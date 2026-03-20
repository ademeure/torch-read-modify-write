"""Reference CUDA kernel for aten.mm — naive nested loop matmul.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_mm.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_mm(
    const float *A, const float *B, float *C,
    unsigned int M, unsigned int K, unsigned int N
) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (unsigned int k = 0; k < K; k++)
            sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}
"""

M, K, N = 64, 32, 48

def init_once():
    A = torch.randn(M, K, device="cuda")
    B = torch.randn(K, N, device="cuda")
    return {
        "kernel_source": KERNEL_SRC, "inputs": [A, B],
        "expected": [torch.ops.aten.mm.default(A, B).flatten()],
        "outputs": ["float32;n=%d" % (M * N)],
        "grid": ((N + 15) // 16, (M + 15) // 16),
        "block": (16, 16), "atol": 1e-3,
    }

def run(inputs, kernel):
    return [kernel(*inputs, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.out_ptr(0),
        np.uint32(M), np.uint32(K), np.uint32(N),
    ])]
