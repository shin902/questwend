#include "ggml.h"

#include <cstdio>

static bool check_shape(int input_width, int rows, int k) {
    ggml_init_params params{};
    params.mem_size = ggml_tensor_overhead() * 16 + ggml_graph_overhead_custom(16, false);
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;

    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input_width, rows);
    ggml_tensor * topk = ggml_cont(ctx, ggml_argsort_top_k(ctx, input, k));
    const bool ok = topk && topk->op == GGML_OP_CONT && topk->ne[0] == k && topk->ne[1] == rows &&
                    topk->src[0] && topk->src[0]->op == GGML_OP_VIEW &&
                    topk->src[0]->src[0] && topk->src[0]->src[0]->op == GGML_OP_ARGSORT;
    ggml_free(ctx);
    return ok;
}

int main() {
    // Flash-Next uses indexer_top_k + compress_ratio - 1 = 2051. This is
    // beyond Vulkan's TOP_K pipeline range and must remain an argsort view.
    if (!check_shape(2560, 512, 2051) || !check_shape(2560, 512, 2050) ||
        !check_shape(2560, 512, 1024)) {
        std::fprintf(stderr, "QSA top-k shape regression failed\n");
        return 1;
    }
    std::puts("QSA top-k shape regression passed");
    return 0;
}
