"""Reference CUDA kernel for aten.addmm — bias + A @ B."""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_addmm(
    const float *bias, const float *A, const float *B, float *C,
    unsigned int M, unsigned int K, unsigned int N
) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = bias[col];
        for (unsigned int k = 0; k < K; k++)
            sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}
"""

M, K, N = 64, 32, 48

def init_once():
    bias = torch.randn(N, device="cuda")
    A = torch.randn(M, K, device="cuda")
    B = torch.randn(K, N, device="cuda")
    return {
        "kernel_source": KERNEL_SRC, "inputs": [bias, A, B],
        "expected": [torch.ops.aten.addmm.default(bias, A, B).flatten()],
        "outputs": ["float32;n=%d" % (M * N)],
        "grid": ((N + 15) // 16, (M + 15) // 16),
        "block": (16, 16), "atol": 1e-3,
    }

def run(inputs, kernel):
    return [kernel(*inputs, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.in_ptr(2), kernel.out_ptr(0),
        np.uint32(M), np.uint32(K), np.uint32(N),
    ])]
