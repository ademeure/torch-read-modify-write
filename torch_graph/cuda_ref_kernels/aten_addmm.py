"""Reference CUDA kernel for aten.addmm — bias + A @ B."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_addmm(
    const float *bias, const float *A, const float *B, float *C,
    unsigned int M, unsigned int K, unsigned int N
) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = bias[col];
        for (unsigned int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor aten_addmm_fwd(torch::Tensor bias, torch::Tensor A, torch::Tensor B) {
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::empty({M, N}, A.options());
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    aten_addmm<<<blocks, threads>>>(bias.data_ptr<float>(), A.data_ptr<float>(),
                                     B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
    return C;
}
"""

def test():
    ext = compile_cuda("aten_addmm", KERNEL_SRC, ["aten_addmm_fwd"])
    bias = torch.randn(48, device="cuda")
    A = torch.randn(64, 32, device="cuda")
    B = torch.randn(32, 48, device="cuda")
    result = ext.aten_addmm_fwd(bias, A.contiguous(), B.contiguous())
    expected = torch.addmm(bias, A, B)
    check("aten.addmm", result, expected, atol=1e-3)
    print("PASS aten.addmm")

if __name__ == "__main__":
    test()
