#!/usr/bin/env python3
"""Replay identical mixed-serving requests serially with MTP and compare useful-work time."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

from mixed_serving_sweep import require_unused_server_port, validate_case, wait_ready


def measure_efficiency(mixed, serial):
    parallel_requests = mixed['requests']
    serial_requests = serial['requests']
    if len(parallel_requests) != len(serial_requests) or not parallel_requests:
        raise ValueError('request count mismatch')
    for parallel, single in zip(parallel_requests, serial_requests):
        for key in ('name', 'prompt', 'requested_output'):
            if parallel[key] != single[key]:
                raise ValueError(f'mismatched {key}')
        if parallel['usage'] != single['usage']:
            raise ValueError('input/output token accounting differs')
    mixed_seconds = max(row['end'] for row in parallel_requests) - min(row['start'] for row in parallel_requests)
    serial_seconds = sum(row['end'] - row['start'] for row in serial_requests)
    if min(mixed_seconds, serial_seconds) <= 0:
        raise ValueError('invalid duration')
    return dict(mixed_seconds=mixed_seconds, serial_mtp_seconds=serial_seconds,
                efficiency=serial_seconds / mixed_seconds,
                input_tokens=sum(row['usage']['prompt_tokens'] for row in parallel_requests),
                output_tokens=sum(row['usage']['completion_tokens'] for row in parallel_requests))


def self_check():
    def row(name, start, end):
        return dict(name=name, prompt=name, requested_output=1, start=start, end=end,
                    usage=dict(prompt_tokens=10, completion_tokens=1, total_tokens=11))
    mixed = dict(requests=[row('decode', 0, 5), row('prefill', 1, 4)])
    serial = dict(requests=[row('decode', 10, 14), row('prefill', 14, 16)])
    assert measure_efficiency(mixed, serial)['efficiency'] == 1.2
    serial['requests'][1]['prompt'] = 'different work'
    try:
        measure_efficiency(mixed, serial)
    except ValueError:
        pass
    else:
        raise AssertionError('unmatched workload accepted')
    print('serial service time / mixed makespan and identical-workload checks passed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--harness-dir', type=Path)
    parser.add_argument('--server', type=Path)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--sources', nargs='+', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--drafts', nargs='+', type=int, default=[1])
    parser.add_argument('--chunk', type=int, default=256)
    parser.add_argument('--port', type=int, default=8218)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if not all([args.harness_dir, args.server, args.model, args.sources, args.output]):
        parser.error('--harness-dir, --server, --model, --sources and --output are required')
    if min([args.chunk] + args.drafts) < 1:
        parser.error('chunk and draft lengths must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    workload = args.harness_dir / 'benchmark.py'
    spec = importlib.util.spec_from_file_location('ornith_workload', workload)
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    harness.MODEL = str(args.model.resolve())
    (args.output / 'workload.py').write_bytes(workload.read_bytes())
    metadata = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    metadata['sources'] = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in args.sources}
    metadata['server_sha256'] = hashlib.sha256(args.server.read_bytes()).hexdigest()
    metadata['uname'] = list(os.uname())
    metadata['runtime_environment'] = {key: value for key, value in os.environ.items()
                                       if key.startswith(('QWEN_', 'GGML_'))}
    (args.output / 'environment.json').write_text(json.dumps(metadata, indent=2))
    comparisons = []
    url = f'http://127.0.0.1:{args.port}'
    require_unused_server_port(url)
    for draft in args.drafts:
        command = [str(args.server.resolve()), '-m', str(args.model.resolve()), '--host', '127.0.0.1',
                   '--port', str(args.port), '--n-ctx', '16384', '--no-mmproj', '--cache-slots', '0',
                   '--pf-chunk', str(args.chunk), '--mtp', '--draft', str(draft)]
        (args.output / f'draft-{draft}-command.json').write_text(json.dumps(command))
        log_path = args.output / f'draft-{draft}-server.log'
        with log_path.open('x') as log:
            server = subprocess.Popen(command, stdout=log, stderr=log)
            try:
                wait_ready(server, url)
                if 'MTP self-speculative decode ON' not in log_path.read_text():
                    raise RuntimeError('server did not confirm MTP is active')
                warmup = harness.run_case(url, 1, 0, 'warmup', 32)
                validate_case(warmup, harness.summarize(warmup))
                (args.output / f'draft-{draft}-warmup.json').write_text(json.dumps(warmup))
                for source in args.sources:
                    mixed = json.loads(source.read_text())
                    mixed_summary = harness.summarize(mixed)
                    validate_case(mixed, mixed_summary)
                    if not mixed.get('prefill') or not any(row['name'].startswith('decode') for row in mixed['requests']):
                        raise ValueError('source must contain overlapping decode and prefill')
                    if not all(row['phases']['during']['chunks'] > 0 for row in mixed_summary['requests']
                               if row['name'].startswith('decode')):
                        raise ValueError('no decoder progress during prefill')
                    name = f'draft-{draft}-{source.stem}'
                    serial = dict(name=name, requests=[])
                    print('START', name, flush=True)
                    for row in mixed['requests']:
                        result = harness.stream_request(url, row['name'], row['prompt'], row['requested_output'])
                        serial['requests'].append(result)
                        (args.output / f'{name}.json').write_text(json.dumps(serial, indent=2))
                    validate_case(serial, harness.summarize(serial))
                    comparison = dict(source=str(source.resolve()), draft=draft, **measure_efficiency(mixed, serial))
                    comparisons.append(comparison)
                    (args.output / 'efficiency.json').write_text(json.dumps(comparisons, indent=2))
                    print(json.dumps(comparison), flush=True)
            finally:
                server.terminate()
                try:
                    server.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
    print('DONE', args.output, flush=True)


if __name__ == '__main__':
    main()
