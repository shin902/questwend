#include "model.h"
#include "runtime.h"
#include "batch_executor.h"
#include "chat.h"
#include "tokenizer.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <filesystem>
#include <random>
#include <stdexcept>

using namespace questwend;

static void check_argmax(bool use_gpu) {
    const auto devices = gpu_devices();
    auto * device = use_gpu && !devices.empty() ? devices.front()
                   : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    auto * backend = ggml_backend_dev_init(device, nullptr);
    if (!backend) throw std::runtime_error("argmax backend initialization failed");
    ggml_init_params params{ggml_tensor_overhead() * 16 + ggml_graph_overhead_custom(16, false), nullptr, true};
    auto * context = ggml_init(params);
    auto * input = ggml_new_tensor_2d(context, GGML_TYPE_F32, 2048, 4);
    ggml_set_input(input);
    auto * output = ggml_argmax(context, input);
    ggml_set_output(output);
    auto * graph = ggml_new_graph_custom(context, 16, false);
    ggml_build_forward_expand(graph, output);
    auto * buffer = ggml_backend_alloc_ctx_tensors(context, backend);
    if (!buffer) {
        ggml_free(context); ggml_backend_free(backend);
        throw std::runtime_error("argmax allocation failed");
    }
    std::vector<float> values(2048 * 4, -2);
    values[1] = values[2] = 1;
    values[2048 + 31] = values[2048 + 32] = 2;
    std::fill(values.begin() + 4096, values.begin() + 6144, -INFINITY);
    values[6144 + 1023] = values[6144 + 1024] = -1;
    ggml_backend_tensor_set(input, values.data(), 0, values.size() * sizeof(float));
    bool passed = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    if (passed) {
        int32_t indices[4];
        ggml_backend_tensor_get(output, indices, 0, sizeof(indices));
        passed = indices[0] == 1 && indices[1] == 31 && indices[2] == 0 && indices[3] == 1023;
    }
    ggml_backend_buffer_free(buffer); ggml_free(context); ggml_backend_free(backend);
    if (!passed) throw std::runtime_error("argmax must preserve first-maximum tie semantics");
}

struct Fixture {
    std::filesystem::path directory;
    ~Fixture() { if (!directory.empty()) { std::error_code error; std::filesystem::remove_all(directory, error); } }
    std::string write(int layers = 4) {
        directory = std::filesystem::temp_directory_path() /
            ("qw-batch-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        if (!std::filesystem::create_directory(directory)) throw std::runtime_error("fixture directory already exists");
        ggml_init_params params{16 * 1024 * 1024, nullptr, false};
        auto * context = ggml_init(params);
        auto * metadata = gguf_init_empty();
        gguf_set_val_str(metadata, "general.architecture", "qwen35moe");
        for (auto field : std::vector<std::pair<const char *, uint32_t>>{
                {"block_count", layers}, {"embedding_length", 128}, {"feed_forward_length", 256},
                {"attention.head_count", 2}, {"attention.head_count_kv", 1},
                {"attention.key_length", 128}, {"context_length", 512},
                {"expert_count", 4}, {"expert_used_count", 2}, {"expert_feed_forward_length", 64},
                {"ssm.inner_size", 512}, {"ssm.group_count", 2}, {"ssm.state_size", 128},
                {"ssm.conv_kernel", 4}, {"ssm.time_step_rank", 4}, {"full_attention_interval", 2}})
            gguf_set_val_u32(metadata, (std::string("qwen35moe.") + field.first).c_str(), field.second);
        const int32_t rope_sections[] = {16, 16, 32, 0};
        gguf_set_arr_data(metadata, "qwen35moe.rope.dimension_sections", GGUF_TYPE_INT32, rope_sections, 4);
        std::mt19937 random(902);
        std::uniform_real_distribution<float> weight(-0.04f, 0.04f);
        auto tensor = [&](const std::string & name, std::vector<int64_t> shape, float fill = NAN) {
            auto * value = ggml_new_tensor(context, GGML_TYPE_F32, shape.size(), shape.data());
            ggml_set_name(value, name.c_str());
            auto * data = (float *) value->data;
            for (int64_t index = 0; index < ggml_nelements(value); ++index)
                data[index] = std::isnan(fill) ? weight(random) : fill;
            gguf_add_tensor(metadata, value);
        };
        tensor("token_embd.weight", {128, 64});
        tensor("output.weight", {128, 64});
        tensor("output_norm.weight", {128}, 1);
        for (int layer = 0; layer < layers; ++layer) {
            const std::string prefix = "blk." + std::to_string(layer) + ".";
            tensor(prefix + "attn_norm.weight", {128}, 1);
            tensor(prefix + "post_attention_norm.weight", {128}, 1);
            if (layer % 2 == 0) {
                tensor(prefix + "attn_qkv.weight", {128, 1024});
                tensor(prefix + "attn_gate.weight", {128, 512});
                tensor(prefix + "ssm_beta.weight", {128, 4});
                tensor(prefix + "ssm_alpha.weight", {128, 4});
                tensor(prefix + "ssm_dt.bias", {4}, 0);
                tensor(prefix + "ssm_a", {4}, -0.5f);
                tensor(prefix + "ssm_conv1d.weight", {4, 1024});
                tensor(prefix + "ssm_norm.weight", {128}, 1);
                tensor(prefix + "ssm_out.weight", {512, 128});
            } else {
                tensor(prefix + "attn_q.weight", {128, 512});
                tensor(prefix + "attn_k.weight", {128, 128});
                tensor(prefix + "attn_v.weight", {128, 128});
                tensor(prefix + "attn_q_norm.weight", {128}, 1);
                tensor(prefix + "attn_k_norm.weight", {128}, 1);
                tensor(prefix + "attn_output.weight", {256, 128});
            }
            tensor(prefix + "ffn_gate_inp.weight", {128, 4});
            tensor(prefix + "ffn_gate_exps.weight", {128, 64, 4});
            tensor(prefix + "ffn_up_exps.weight", {128, 64, 4});
            tensor(prefix + "ffn_down_exps.weight", {64, 128, 4});
        }
        const auto path = (directory / "model.gguf").string();
        const bool written = gguf_write_to_file(metadata, path.c_str(), false);
        gguf_free(metadata);
        ggml_free(context);
        if (!written) throw std::runtime_error("fixture write failed");
        return path;
    }
};

static bool require_matching_logits(const std::vector<float> & serial,
                                    const Runtime::BatchOutput & batched, int past) {
    if (batched.n_past != past || serial.size() != batched.logits.size())
        throw std::runtime_error("batch position or vocabulary size mismatch");
    double maximum = 0, squared_error = 0, serial_norm = 0, batch_norm = 0, dot = 0;
    for (size_t token = 0; token < serial.size(); ++token) {
        const double delta = std::fabs(serial[token] - batched.logits[token]);
        maximum = std::max(maximum, delta);
        squared_error += delta * delta;
        serial_norm += double(serial[token]) * serial[token];
        batch_norm += double(batched.logits[token]) * batched.logits[token];
        dot += double(serial[token]) * batched.logits[token];
        if (!std::isfinite(batched.logits[token]))
            throw std::runtime_error("non-finite batched logits");
    }
    const auto serial_token = std::max_element(serial.begin(), serial.end()) - serial.begin();
    const auto batch_token = std::max_element(batched.logits.begin(), batched.logits.end()) - batched.logits.begin();
    std::printf("position=%d max_difference=%.6f rms_difference=%.6f cosine=%.9f serial_token=%td batch_token=%td\n",
                past, maximum, std::sqrt(squared_error / serial.size()),
                dot / std::sqrt(serial_norm * batch_norm), serial_token, batch_token);
    return maximum <= 0.001 && serial_token == batch_token;
}

int main(int argc, char ** argv) {
    if (argc > 3 || (argc == 3 && std::strcmp(argv[2], "--cpu") != 0)) {
        std::fprintf(stderr, "usage: %s [model.gguf | --gdn-tail] [--cpu]\n", argv[0]);
        return 2;
    }
    try {
        Fixture fixture;
        const bool gdn_tail = argc >= 2 && std::strcmp(argv[1], "--gdn-tail") == 0;
        const bool synthetic = argc == 1 || gdn_tail || std::strcmp(argv[1], "--cpu") == 0;
        auto model = Model::load(synthetic ? fixture.write(gdn_tail ? 3 : 4) : argv[1]);
        Tokenizer tokenizer(model->vocab());
        RuntimeConfig config;
        config.n_ctx = 512;
        config.use_cuda = !(argc >= 2 && std::strcmp(argv[argc - 1], "--cpu") == 0);
        Runtime runtime(*model, config);
        std::vector<std::vector<int32_t>> prompts;
        if (synthetic) prompts = {{1, 3, 5, 7, 9}, {2, 4, 6, 8, 10, 12, 14},
                                  {31, 29, 27, 25}, {32, 30, 28, 26, 24, 22}};
        else for (const auto * text : {
                "The ocean is blue. Explain how sunlight reaches the deep sea.",
                "Write a concise explanation of concurrency and memory isolation in a server.",
                "A small bird sits on the windowsill and sings every morning.",
                "Compute the sum of three, seven, and eleven. Show each step clearly."})
            prompts.push_back(build_chatml_tokens(tokenizer, {{"user", text}}, true, false));
        const auto continuation = synthetic ? std::vector<int32_t>{11, 13, 15, 17}
                                           : tokenizer.encode(" Then consider the next observation.");
        std::vector<std::vector<float>> expected, continued;
        std::vector<std::vector<std::vector<float>>> decode_steps;
        std::vector<std::vector<int32_t>> step_tokens;
        for (const auto & prompt : prompts) {
            runtime.reset();
            expected.push_back(runtime.decode(prompt));
            continued.push_back(runtime.decode(continuation));
            runtime.reset();
            auto logits = runtime.decode(prompt);
            std::vector<std::vector<float>> steps;
            std::vector<int32_t> tokens;
            for (int step = 0; step < 4; ++step) {
                const int32_t token = synthetic ? continuation[step % continuation.size()]
                    : (int32_t) (std::max_element(logits.begin(), logits.end()) - logits.begin());
                tokens.push_back(token);
                logits = runtime.decode({token});
                steps.push_back(logits);
            }
            decode_steps.push_back(std::move(steps));
            step_tokens.push_back(std::move(tokens));
        }
        if (!synthetic) {
            for (int slot = 0; slot < 4; ++slot) {
                runtime.reset();
                std::vector<float> split_logits;
                for (auto token : prompts[slot]) split_logits = runtime.decode({token});
                std::printf("SERIAL_CHUNKING slot=%d ", slot);
                require_matching_logits(expected[slot], {split_logits, (int) prompts[slot].size()}, prompts[slot].size());
            }
        }
        runtime.configure_batch_slots(4);
        bool matched = true;
        auto isolated_slot = runtime.decode_batch({{1, prompts[1], true}});
        matched &= require_matching_logits(expected[1], isolated_slot[0], prompts[1].size());
        auto isolated_step = runtime.decode_batch({{1, {step_tokens[1][0]}}});
        std::printf("ISOLATED_DECODE slot=1 ");
        matched &= require_matching_logits(decode_steps[1][0], isolated_step[0], prompts[1].size() + 1);
        std::vector<Runtime::BatchInput> inputs;
        for (int slot = 0; slot < 4; ++slot) inputs.push_back({slot, prompts[slot], true});
        auto outputs = runtime.decode_batch(inputs);
        auto departing_peers = runtime.decode_batch({{1, {step_tokens[1][0]}}});
        std::printf("PACKED_PREFILL_SINGLE_DECODE slot=1 ");
        matched &= require_matching_logits(decode_steps[1][0], departing_peers[0], prompts[1].size() + 1);
        auto neighbours = inputs;
        for (int slot : {0, 2, 3})
            std::reverse(neighbours[slot].tokens.begin(), neighbours[slot].tokens.end());
        auto independent = runtime.decode_batch(neighbours);
        std::printf("NEIGHBOUR_CHANGE slot=1 ");
        matched &= require_matching_logits(outputs[1].logits, independent[1], prompts[1].size());
        runtime.decode_batch(inputs);
        for (int slot = 0; slot < 4; ++slot)
            matched &= require_matching_logits(expected[slot], outputs[slot], prompts[slot].size());
        // Ragged mixed round: one sequence advances one token, another prefills
        // several; neither may see the other's KV or recurrent-state updates.
        auto mixed = runtime.decode_batch({{0, {continuation[0]}}, {1, continuation}});
        matched &= require_matching_logits(continued[1], mixed[1], prompts[1].size() + continuation.size());
        std::vector<int32_t> tail(continuation.begin() + 1, continuation.end());
        auto completed = runtime.decode_batch({{0, tail}});
        matched &= require_matching_logits(continued[0], completed[0], prompts[0].size() + continuation.size());
        auto reused = runtime.decode_batch({{0, prompts[2], true}, {1, prompts[3], true}});
        matched &= require_matching_logits(expected[2], reused[0], prompts[2].size());
        matched &= require_matching_logits(expected[3], reused[1], prompts[3].size());
        bool rejected = false;
        try { runtime.decode_batch({{0, continuation}, {0, continuation}}); }
        catch (const std::runtime_error &) { rejected = true; }
        if (!rejected) throw std::runtime_error("duplicate slot accepted");
        auto intact = runtime.decode_batch({{0, continuation}});
        matched &= require_matching_logits(continued[2], intact[0], prompts[2].size() + continuation.size());
        runtime.decode_batch(inputs);
        for (int step = 0; step < 4; ++step) {
            std::vector<Runtime::BatchInput> round;
            for (int slot = 0; slot < 4; ++slot)
                round.push_back({slot, {step_tokens[slot][step]}});
            auto decoded = runtime.decode_batch(round);
            for (int slot = 0; slot < 4; ++slot)
                matched &= require_matching_logits(decode_steps[slot][step], decoded[slot], prompts[slot].size() + step + 1);
        }
        if (!matched) throw std::runtime_error("batched logits or greedy token diverged from serial");
        for (auto & input : inputs) input.greedy = true;
        auto greedy = runtime.decode_batch(inputs);
        for (int slot = 0; slot < 4; ++slot) {
            const auto & logits = outputs[slot].logits;
            const int32_t expected_token = (int32_t) (std::max_element(logits.begin(), logits.end()) - logits.begin());
            if (greedy[slot].greedy_token != expected_token || !greedy[slot].logits.empty() ||
                greedy[slot].n_past != (int) prompts[slot].size())
                throw std::runtime_error("device greedy selection differs from logits sampling");
        }
        for (auto & input : inputs) input.want_logits = false;
        auto silent = runtime.decode_batch(inputs);
        for (int slot = 0; slot < 4; ++slot)
            if (!silent[slot].logits.empty() || silent[slot].greedy_token != -1 ||
                silent[slot].n_past != (int) prompts[slot].size())
                throw std::runtime_error("state-only prefill returned a readout or wrong position");
        auto readout = runtime.decode_batch({{0, continuation}, {1, continuation}});
        matched &= require_matching_logits(continued[0], readout[0], prompts[0].size() + continuation.size());
        matched &= require_matching_logits(continued[1], readout[1], prompts[1].size() + continuation.size());
        auto selected = runtime.decode_batch({{0, prompts[0], true, true, false},
                                             {1, prompts[1], true, true, true}});
        if (!selected[0].logits.empty() || selected[0].greedy_token != -1 ||
                selected[1].greedy_token != greedy[1].greedy_token)
            throw std::runtime_error("selective readout lost its row ownership");
        if (!matched) throw std::runtime_error("state-only prefill corrupted subsequent outputs");
        check_argmax(config.use_cuda);
        if (synthetic) {
            auto queued_model = Model::load((fixture.directory / "model.gguf").string());
            Runtime queued_runtime(*queued_model, config);
            BatchExecutor executor(queued_runtime, 4, 3, 2);
            std::vector<std::future<bool>> clients;
            for (int slot = 0; slot < 4; ++slot) {
                clients.push_back(std::async(std::launch::async, [&, slot] {
                    auto lease = executor.acquire_for(std::chrono::seconds(30));
                    if (!lease) throw std::runtime_error("slot acquisition timed out");
                    auto output = executor.decode(*lease, prompts[slot]);
                    bool equal = require_matching_logits(expected[slot], output, prompts[slot].size());
                    for (int step = 0; step < 4; ++step) {
                        output = executor.decode(*lease, {step_tokens[slot][step]});
                        equal &= require_matching_logits(decode_steps[slot][step], output, prompts[slot].size() + step + 1);
                    }
                    return equal;
                }));
            }
            for (auto & client : clients)
                if (!client.get()) throw std::runtime_error("queued client state diverged");
            auto cancelled = executor.acquire_for(std::chrono::seconds(1));
            if (!cancelled) throw std::runtime_error("cancelled lease acquisition failed");
            cancelled->cancel();
            bool cancel_rejected = false;
            try { executor.decode(*cancelled, prompts[0]); }
            catch (const std::runtime_error &) { cancel_rejected = true; }
            if (!cancel_rejected) throw std::runtime_error("cancelled request accepted");
            cancelled.reset();
            auto reused_lease = executor.acquire_for(std::chrono::seconds(1));
            if (!reused_lease) throw std::runtime_error("slot was not released after cancellation");
            auto final = executor.decode(*reused_lease, prompts[0], true);
            if (final.greedy_token != greedy[0].greedy_token || final.n_past != (int) prompts[0].size())
                throw std::runtime_error("cancelled slot was not safely reusable");
        }
        std::puts("independent slots, mixed rounds, slot reuse, validation and queued clients passed");
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
