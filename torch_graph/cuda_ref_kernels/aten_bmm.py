"""Reference CUDA kernel for aten.bmm — batched matmul."""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_bmm(
    const float *A, const float *B, float *C,
    unsigned int batch, unsigned int M, unsigned int K, unsigned int N
) {
    unsigned int b = blockIdx.z;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch && row < M && col < N) {
        float sum = 0.0f;
        for (unsigned int k = 0; k < K; k++)
            sum += A[b*M*K + row*K + k] * B[b*K*N + k*N + col];
        C[b*M*N + row*N + col] = sum;
    }
}
"""

B, M, K, N = 4, 16, 32, 24

def init_once():
    A = torch.randn(B, M, K, device="cuda")
    Bt = torch.randn(B, K, N, device="cuda")
    return {
        "kernel_source": KERNEL_SRC, "inputs": [A, Bt],
        "expected": [torch.ops.aten.bmm.default(A, Bt).flatten()],
        "outputs": ["float32;n=%d" % (B * M * N)],
        "grid": ((N + 15) // 16, (M + 15) // 16, B),
        "block": (16, 16), "atol": 1e-3,
    }

def run(inputs, kernel):
    return [kernel(*inputs, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.out_ptr(0),
        np.uint32(B), np.uint32(M), np.uint32(K), np.uint32(N),
    ])]
