"""Reference CUDA kernel for aten.outer — outer product of two vectors.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_outer.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_outer(
    const float *a, const float *b, float *out,
    unsigned int M, unsigned int N
) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N)
        out[row * N + col] = a[row] * b[col];
}
"""

M, N = 64, 48

def init_once():
    a = torch.randn(M, device="cuda")
    b = torch.randn(N, device="cuda")
    return {
        "kernel_source": KERNEL_SRC, "inputs": [a, b],
        "expected": [torch.ops.aten.outer.default(a, b).flatten()],
        "outputs": ["float32;n=%d" % (M * N)],
        "grid": ((N + 15) // 16, (M + 15) // 16),
        "block": (16, 16), "atol": 1e-5,
    }

def run(inputs, kernel):
    return [kernel(*inputs, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.out_ptr(0),
        np.uint32(M), np.uint32(N),
    ])]
