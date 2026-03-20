"""Reference CUDA kernel for aten.scatter — scatter values into tensor by index."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

// Scatter along dim=1: out = input.clone(); out[i][index[i][j]] = src[i][j]
// Naive: one thread per src element, atomic write
extern "C" __global__ void aten_scatter_2d(
    const float *input, const long *index, const float *src, float *output,
    unsigned int rows, unsigned int in_cols, unsigned int src_cols
) {
    // First copy input → output
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_in = rows * in_cols;
    if (idx < total_in) output[idx] = input[idx];
}

extern "C" __global__ void aten_scatter_write(
    const long *index, const float *src, float *output,
    unsigned int rows, unsigned int in_cols, unsigned int src_cols
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_src = rows * src_cols;
    if (idx >= total_src) return;
    unsigned int r = idx / src_cols, c = idx % src_cols;
    long dst_c = index[idx];
    output[r * in_cols + dst_c] = src[idx];
}

torch::Tensor aten_scatter_2d_fwd(
    torch::Tensor input, torch::Tensor index, torch::Tensor src
) {
    auto ci = input.contiguous();
    int rows = ci.size(0), in_cols = ci.size(1), src_cols = index.size(1);
    auto output = torch::empty_like(ci);
    int total_in = rows * in_cols;
    aten_scatter_2d<<<(total_in+255)/256, 256>>>(
        ci.data_ptr<float>(), index.data_ptr<long>(), src.data_ptr<float>(),
        output.data_ptr<float>(), rows, in_cols, src_cols);
    int total_src = rows * src_cols;
    aten_scatter_write<<<(total_src+255)/256, 256>>>(
        index.data_ptr<long>(), src.data_ptr<float>(),
        output.data_ptr<float>(), rows, in_cols, src_cols);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_scatter_2d", KERNEL_SRC, ["aten_scatter_2d_fwd"])
    x = torch.zeros(8, 32, device="cuda")
    idx = torch.randint(0, 32, (8, 16), device="cuda")
    src = torch.randn(8, 16, device="cuda")
    result = ext.aten_scatter_2d_fwd(x, idx, src)
    expected = aten.scatter.src(x, 1, idx, src)
    check("aten.scatter", result, expected)
    print("PASS aten.scatter")

if __name__ == "__main__":
    test()
