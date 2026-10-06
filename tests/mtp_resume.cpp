#include "chat.h"
#include "model.h"
#include "runtime.h"
#include "tokenizer.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <utility>

using namespace questwend;

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3 || (argc == 3 && std::strcmp(argv[2], "--cpu"))) {
        std::fprintf(stderr, "usage: %s model.gguf [--cpu]\n", argv[0]);
        return 2;
    }
    try {
        auto model = Model::load(argv[1]);
        Tokenizer tokenizer(model->vocab());
        RuntimeConfig config;
        config.n_ctx = 16384;
        config.use_cuda = argc != 3;
        config.use_mtp = true;
        Runtime runtime(*model, config);
        if (!runtime.has_mtp()) throw std::runtime_error("model must support MTP");
        const auto prompt = build_chatml_tokens(tokenizer, {{"user",
            "1e4e341426bb45ec9d597874d5950ac4\n"
            "Write a very long numbered list of 1000 detailed engineering observations. "
            "Each item must contain at least two sentences. Start immediately and continue "
            "without a conclusion. Do not summarize or stop early."}}, true, false);
        constexpr int budget = 768;
        std::vector<int32_t> uninterrupted;
        runtime.generate_mtp(prompt, budget, 1, [&](int32_t token) {
            uninterrupted.push_back(token);
            return true;
        });
        if (uninterrupted.size() != budget) throw std::runtime_error("uninterrupted output budget mismatch");
        const auto other_prompt = build_chatml_tokens(tokenizer, {{"user", "Explain tides and ocean currents."}}, true, false);
        for (const auto &[slice, switch_context] : {std::pair{1, false}, {7, false}, {16, false}, {16, true}}) {
            runtime.reset();
            std::vector<int32_t> resumed;
            int32_t pending = -1;
            auto tail = prompt;
            while (resumed.size() < budget) {
                const auto &confirmed = runtime.kv_tokens();
                while (prompt.size() + resumed.size() < confirmed.size() && resumed.size() < budget)
                    resumed.push_back(confirmed[prompt.size() + resumed.size()]);
                if (resumed.size() == budget) break;
                const int32_t resume_token = pending;
                if (resume_token >= 0 && switch_context) {
                    std::vector<uint8_t> saved;
                    saved.reserve(runtime.state_bytes());
                    runtime.save_state([&](const void *bytes, size_t count) {
                        const auto *begin = static_cast<const uint8_t *>(bytes);
                        saved.insert(saved.end(), begin, begin + count);
                    });
                    runtime.reset();
                    runtime.prefill(other_prompt, true);
                    size_t offset = 0;
                    runtime.load_state([&](void *bytes, size_t count) {
                        if (offset + count > saved.size()) throw std::runtime_error("truncated saved state");
                        std::memcpy(bytes, saved.data() + offset, count);
                        offset += count;
                    });
                }
                pending = -1;
                int offered = 0;
                runtime.generate_mtp(tail, budget - resumed.size(), 1, [&](int32_t token) {
                    if (offered == slice) return false;
                    resumed.push_back(token);
                    ++offered;
                    return true;
                }, &pending, false, resume_token);
                tail.clear();
                if (pending < 0 && resumed.size() < budget)
                    throw std::runtime_error("paused generation lost its pending token");
            }
            const auto difference = std::mismatch(uninterrupted.begin(), uninterrupted.end(), resumed.begin());
            const auto position = difference.first - uninterrupted.begin();
            std::printf("slice=%d switch_context=%d tokens=%zu first_difference=%td\n", slice, switch_context, resumed.size(), position);
            if (position != budget) throw std::runtime_error("resuming MTP must preserve uninterrupted tokens");
        }
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
