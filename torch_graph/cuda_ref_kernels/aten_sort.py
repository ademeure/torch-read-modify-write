"""Reference CUDA kernel for aten.sort — bubble sort reference (intentionally slow)."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

// One block per row, single-thread bubble sort (correct but slow reference)
extern "C" __global__ void aten_sort_kernel(
    const float *input, float *values, long *indices,
    unsigned int rows, unsigned int cols
) {
    unsigned int row = blockIdx.x;
    if (row >= rows) return;
    const float *ri = input + row * cols;
    float *rv = values + row * cols;
    long *ri_idx = indices + row * cols;
    // Init
    for (unsigned int j = 0; j < cols; j++) { rv[j] = ri[j]; ri_idx[j] = j; }
    // Bubble sort ascending
    for (unsigned int i = 0; i < cols; i++) {
        for (unsigned int j = i + 1; j < cols; j++) {
            if (rv[j] < rv[i]) {
                float tmp = rv[i]; rv[i] = rv[j]; rv[j] = tmp;
                long ti = ri_idx[i]; ri_idx[i] = ri_idx[j]; ri_idx[j] = ti;
            }
        }
    }
}

std::vector<torch::Tensor> aten_sort_fwd(torch::Tensor input) {
    int cols = input.size(-1);
    int rows = input.numel() / cols;
    auto flat = input.reshape({rows, cols}).contiguous();
    auto values = torch::empty_like(flat);
    auto indices = torch::empty({rows, cols}, torch::TensorOptions().dtype(torch::kLong).device(input.device()));
    aten_sort_kernel<<<rows, 1>>>(
        flat.data_ptr<float>(), values.data_ptr<float>(), indices.data_ptr<long>(), rows, cols);
    return {values.reshape(input.sizes()), indices.reshape(input.sizes())};
}
"""

def test():
    ext = compile_cuda("aten_sort_kernel", KERNEL_SRC, ["aten_sort_fwd"])
    x = torch.randn(8, 32, device="cuda")
    result = ext.aten_sort_fwd(x)
    expected = aten.sort.default(x, -1)
    check("aten.sort.values", result[0], expected[0])
    check("aten.sort.indices", result[1], expected[1])
    print("PASS aten.sort")

if __name__ == "__main__":
    test()
