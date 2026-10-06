#!/usr/bin/env python3
"""Run the existing Ornith workload with matched isolated and mixed baselines."""
import argparse
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request


def measure_retention(isolated_decode, isolated_prefill, mixed):
    decoders = [row for row in mixed['requests'] if row['name'].startswith('decode')]
    during = sum(row['phases']['during']['chunk_tps'] for row in decoders)
    return dict(decode_retention=during / isolated_decode,
                prefill_retention=mixed['effective_prefill_tps'] / isolated_prefill,
                mixed_decode_chunks_s=during,
                prefill_ttft_s=mixed['prefill_ttft_s'],
                worst_gap_s=max(row['phases']['during']['no_arrival_max_s']
                                for row in decoders))


def validate_case(case, summary):
    for request in case['requests']:
        if request.get('error') or not request['events']:
            raise RuntimeError(f"request failed: {request['name']}: {request.get('error')}")
        if request.get('usage', {}).get('completion_tokens') != request['requested_output']:
            raise RuntimeError(f"output budget not reached: {request['name']}")
        if request.get('timings', {}).get('cached_tokens') != 0:
            raise RuntimeError(f"coldness not confirmed: {request['name']}")
    if any(row.get('decode_survived_prefill') is False for row in summary['requests']):
        raise RuntimeError('decoder ended before the recovery window completed')


def require_unused_server_port(url):
    try:
        urllib.request.urlopen(url + '/health', timeout=1).close()
    except OSError:
        return
    raise RuntimeError('port already has a server; refusing contaminated measurement')


def wait_ready(server, url):
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f'server exited: {server.returncode}')
        try:
            with urllib.request.urlopen(url + '/health', timeout=1) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(1)
    raise TimeoutError('server did not become ready')


def self_check():
    mixed = dict(effective_prefill_tps=350, prefill_ttft_s=12,
                 requests=[dict(name='decode-0', phases=dict(during=dict(
                     chunk_tps=21, no_arrival_max_s=0.6))),
                           dict(name='decode-1', phases=dict(during=dict(
                     chunk_tps=14, no_arrival_max_s=0.9)))])
    retention = measure_retention(50, 1000, mixed)
    assert retention['decode_retention'] == 0.7
    assert retention['prefill_retention'] == 0.35
    assert retention['worst_gap_s'] == 0.9
    print('aggregate decode and isolated prefill denominator check passed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--harness-dir', type=Path)
    parser.add_argument('--server', type=Path)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--chunks', default='256,512,1024,2048')
    parser.add_argument('--slice', type=int, default=16)
    parser.add_argument('--batch-slots', type=int, default=0)
    parser.add_argument('--mixed-output', type=int, default=2048)
    parser.add_argument('--decode-output', type=int, default=384)
    parser.add_argument('--mtp', action='store_true')
    parser.add_argument('--cases', nargs='+', choices=['b1', 'b2', 'b4', 'p12', 'm1k12', 'm2k12'],
                        default=['b1', 'b2', 'b4', 'p12', 'm1k12', 'm2k12'])
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--port', type=int, default=8218)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if not all([args.harness_dir, args.server, args.model, args.output]):
        parser.error('--harness-dir, --server, --model and --output are required')
    chunks = [int(chunk) for chunk in args.chunks.split(',')]
    if args.mtp and args.batch_slots:
        parser.error('MTP and experimental batch slots cannot currently be combined')
    if min(chunks) < 1 or min(args.repeats, args.slice, args.decode_output, args.mixed_output) < 1:
        parser.error('chunk, repeats and slice must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    harness_path = args.harness_dir.resolve() / 'benchmark.py'
    spec = importlib.util.spec_from_file_location('ornith_workload', harness_path)
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    harness.MODEL = str(args.model.resolve())
    (args.output / 'workload.py').write_bytes(harness_path.read_bytes())
    url = f'http://127.0.0.1:{args.port}'
    require_unused_server_port(url)
    metadata = {key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()}
    metadata['server'] = str(args.server.resolve())
    metadata['model'] = str(args.model.resolve())
    metadata['uname'] = list(os.uname())
    metadata['server_sha256'] = hashlib.sha256(args.server.read_bytes()).hexdigest()
    metadata['runtime_environment'] = {key: value for key, value in os.environ.items()
                                       if key.startswith(('QWEN_', 'GGML_'))}
    (args.output / 'environment.json').write_text(json.dumps(metadata, indent=2))
    comparisons = []
    for chunk in chunks:
        command = [str(args.server.resolve()), '-m', str(args.model.resolve()),
                   '--host', '127.0.0.1', '--port', str(args.port), '--n-ctx', '16384',
                   '--no-mmproj', '--cache-slots', '4', '--time-slice', str(args.slice),
                   '--pf-chunk', str(chunk)]
        if args.batch_slots:
            command += ['--batch-slots', str(args.batch_slots)]
        if args.mtp:
            command += ['--mtp']
        (args.output / f'chunk-{chunk}-command.json').write_text(json.dumps(command))
        with (args.output / f'chunk-{chunk}-server.log').open('x') as log:
            server = subprocess.Popen(command, stdout=log, stderr=log)
            try:
                wait_ready(server, url)
                warmup = harness.run_case(url, 1, 0, 'warmup', 32)
                (args.output / f'chunk-{chunk}-warmup.json').write_text(json.dumps(warmup))
                validate_case(warmup, harness.summarize(warmup))
                for repetition in range(args.repeats):
                    summaries = {}
                    for label, decoders, prefill, budget in [
                        ('b1', 1, 0, args.decode_output), ('b2', 2, 0, args.decode_output),
                        ('b4', 4, 0, args.decode_output),
                        ('p12', 0, 12288, 1), ('m1k12', 1, 12288, args.mixed_output),
                        ('m2k12', 2, 12288, args.mixed_output)]:
                        if label not in args.cases:
                            continue
                        name = f'chunk-{chunk}-{label}-r{repetition}'
                        print('START', name, flush=True)
                        case = harness.run_case(url, decoders, prefill, name, budget)
                        summary = harness.summarize(case)
                        (args.output / f'{name}.json').write_text(json.dumps(case, indent=2))
                        (args.output / f'{name}-summary.json').write_text(json.dumps(summary, indent=2))
                        validate_case(case, summary)
                        summaries[label] = summary
                        print(json.dumps(summary), flush=True)
                    for count in (1, 2):
                        if not {'p12', f'b{count}', f'm{count}k12'} <= summaries.keys():
                            continue
                        isolated_prefill = summaries['p12']['effective_prefill_tps']
                        baseline_case = json.loads((args.output / f'chunk-{chunk}-b{count}-r{repetition}.json').read_text())
                        start = min(row['events'][0]['t'] for row in baseline_case['requests'])
                        finish = max(row['events'][-1]['t'] for row in baseline_case['requests'])
                        # Same chunk-arrival unit as mixed; usage TPS is not a chunk rate.
                        isolated_decode = sum(len(row['events']) - 1 for row in baseline_case['requests']) / (finish - start)
                        comparison = dict(chunk=chunk, repetition=repetition, decoders=count,
                                          **measure_retention(isolated_decode, isolated_prefill,
                                                              summaries[f'm{count}k12']))
                        comparisons.append(comparison)
                        print('RETENTION', json.dumps(comparison), flush=True)
                    (args.output / 'retention.json').write_text(json.dumps(comparisons, indent=2))
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
