"""Reference CUDA kernel for aten.outer — outer product of two vectors."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_outer(
    const float *a, const float *b, float *out,
    unsigned int M, unsigned int N
) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N)
        out[row * N + col] = a[row] * b[col];
}

torch::Tensor aten_outer_fwd(torch::Tensor a, torch::Tensor b) {
    int M = a.numel(), N = b.numel();
    auto out = torch::empty({M, N}, a.options());
    dim3 threads(16, 16);
    dim3 blocks((N+15)/16, (M+15)/16);
    aten_outer<<<blocks, threads>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), M, N);
    return out;
}
"""

def test():
    ext = compile_cuda("aten_outer", KERNEL_SRC, ["aten_outer_fwd"])
    a = torch.randn(64, device="cuda")
    b = torch.randn(48, device="cuda")
    result = ext.aten_outer_fwd(a, b)
    expected = aten.outer.default(a, b)
    check("aten.outer", result, expected, atol=1e-5)
    print("PASS aten.outer")

if __name__ == "__main__":
    test()
