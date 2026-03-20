"""Reference CUDA kernel for aten.baddbmm — batch add + batch matmul."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_baddbmm(
    const float *self, const float *A, const float *B, float *out,
    unsigned int batch, unsigned int M, unsigned int K, unsigned int N,
    float beta, float alpha
) {
    unsigned int b = blockIdx.z;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch && row < M && col < N) {
        unsigned int off = b * M * N + row * N + col;
        float sum = 0.0f;
        for (unsigned int k = 0; k < K; k++)
            sum += A[b*M*K + row*K + k] * B[b*K*N + k*N + col];
        out[off] = beta * self[off] + alpha * sum;
    }
}

torch::Tensor aten_baddbmm_fwd(
    torch::Tensor self, torch::Tensor A, torch::Tensor B, double beta, double alpha
) {
    int batch = A.size(0), M = A.size(1), K = A.size(2), N = B.size(2);
    auto out = torch::empty({batch, M, N}, A.options());
    dim3 threads(16, 16);
    dim3 blocks((N+15)/16, (M+15)/16, batch);
    aten_baddbmm<<<blocks, threads>>>(
        self.data_ptr<float>(), A.data_ptr<float>(), B.data_ptr<float>(),
        out.data_ptr<float>(), batch, M, K, N, (float)beta, (float)alpha);
    return out;
}
"""

def test():
    ext = compile_cuda("aten_baddbmm", KERNEL_SRC, ["aten_baddbmm_fwd"])
    self = torch.randn(4, 16, 24, device="cuda")
    A = torch.randn(4, 16, 32, device="cuda")
    B = torch.randn(4, 32, 24, device="cuda")
    result = ext.aten_baddbmm_fwd(self.contiguous(), A.contiguous(), B.contiguous(), 1.0, 1.0)
    expected = aten.baddbmm.default(self, A, B)
    check("aten.baddbmm", result, expected, atol=1e-3)
    print("PASS aten.baddbmm")

if __name__ == "__main__":
    test()
