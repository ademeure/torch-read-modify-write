"""Reference CUDA kernel for aten.bmm — batched matmul, nested loops."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_bmm(
    const float *A, const float *B, float *C,
    unsigned int batch, unsigned int M, unsigned int K, unsigned int N
) {
    unsigned int b = blockIdx.z;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch && row < M && col < N) {
        const float *Ab = A + b * M * K;
        const float *Bb = B + b * K * N;
        float *Cb = C + b * M * N;
        float sum = 0.0f;
        for (unsigned int k = 0; k < K; k++) {
            sum += Ab[row * K + k] * Bb[k * N + col];
        }
        Cb[row * N + col] = sum;
    }
}

torch::Tensor aten_bmm_fwd(torch::Tensor A, torch::Tensor B) {
    int batch = A.size(0), M = A.size(1), K = A.size(2), N = B.size(2);
    auto C = torch::empty({batch, M, N}, A.options());
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16, batch);
    aten_bmm<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(),
                                   C.data_ptr<float>(), batch, M, K, N);
    return C;
}
"""

def test():
    ext = compile_cuda("aten_bmm", KERNEL_SRC, ["aten_bmm_fwd"])
    A = torch.randn(4, 16, 32, device="cuda")
    B = torch.randn(4, 32, 24, device="cuda")
    result = ext.aten_bmm_fwd(A.contiguous(), B.contiguous())
    expected = aten.bmm.default(A, B)
    check("aten.bmm", result, expected, atol=1e-3)
    print("PASS aten.bmm")

if __name__ == "__main__":
    test()
