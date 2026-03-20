"""Reference CUDA kernel for aten.mv — matrix-vector multiply."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_mv(
    const float *A, const float *x, float *y,
    unsigned int M, unsigned int K
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        float sum = 0.0f;
        for (unsigned int k = 0; k < K; k++)
            sum += A[row * K + k] * x[k];
        y[row] = sum;
    }
}

torch::Tensor aten_mv_fwd(torch::Tensor A, torch::Tensor x) {
    int M = A.size(0), K = A.size(1);
    auto y = torch::empty({M}, A.options());
    aten_mv<<<(M+255)/256, 256>>>(
        A.data_ptr<float>(), x.data_ptr<float>(), y.data_ptr<float>(), M, K);
    return y;
}
"""

def test():
    ext = compile_cuda("aten_mv", KERNEL_SRC, ["aten_mv_fwd"])
    A = torch.randn(64, 32, device="cuda")
    x = torch.randn(32, device="cuda")
    result = ext.aten_mv_fwd(A.contiguous(), x.contiguous())
    expected = aten.mv.default(A, x)
    check("aten.mv", result, expected, atol=1e-3)
    print("PASS aten.mv")

if __name__ == "__main__":
    test()
