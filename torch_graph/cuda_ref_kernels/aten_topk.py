"""Reference CUDA kernel for aten.topk — find k largest values."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

// One block per row, selection sort for top-k (reference, not optimized)
extern "C" __global__ void aten_topk_kernel(
    const float *input, float *values, long *indices,
    unsigned int rows, unsigned int cols, unsigned int k
) {
    unsigned int row = blockIdx.x;
    if (row >= rows) return;
    const float *ri = input + row * cols;
    float *rv = values + row * k;
    long *ri_idx = indices + row * k;
    // Selection: find k largest
    for (unsigned int i = 0; i < k; i++) {
        float best = -1e38f;
        long best_j = 0;
        for (unsigned int j = 0; j < cols; j++) {
            float v = ri[j];
            bool already = false;
            for (unsigned int p = 0; p < i; p++) {
                if (ri_idx[p] == (long)j) { already = true; break; }
            }
            if (!already && v > best) { best = v; best_j = j; }
        }
        rv[i] = best;
        ri_idx[i] = best_j;
    }
}

std::vector<torch::Tensor> aten_topk_fwd(torch::Tensor input, int64_t k) {
    int cols = input.size(-1);
    int rows = input.numel() / cols;
    auto flat = input.reshape({rows, cols}).contiguous();
    auto values = torch::empty({rows, k}, flat.options());
    auto indices = torch::empty({rows, k}, torch::TensorOptions().dtype(torch::kLong).device(input.device()));
    aten_topk_kernel<<<rows, 1>>>(
        flat.data_ptr<float>(), values.data_ptr<float>(), indices.data_ptr<long>(), rows, cols, k);
    return {values, indices};
}
"""

def test():
    ext = compile_cuda("aten_topk_kernel", KERNEL_SRC, ["aten_topk_fwd"])
    x = torch.randn(8, 32, device="cuda")
    result = ext.aten_topk_fwd(x, 5)
    expected = aten.topk.default(x, 5, -1)
    check("aten.topk.values", result[0], expected[0])
    check("aten.topk.indices", result[1], expected[1])
    print("PASS aten.topk")

if __name__ == "__main__":
    test()
