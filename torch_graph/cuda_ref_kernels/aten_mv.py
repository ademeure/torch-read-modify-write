"""Reference CUDA kernel for aten.mv — matrix-vector multiply.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_mv.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
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
"""

M, K = 64, 32

def init_once():
    A = torch.randn(M, K, device="cuda")
    x = torch.randn(K, device="cuda")
    return {
        "kernel_source": KERNEL_SRC, "inputs": [A, x],
        "expected": [torch.ops.aten.mv.default(A, x)],
        "outputs": ["float32;n=%d" % M],
        "grid": ((M + 255) // 256,), "atol": 1e-3,
    }

def run(inputs, kernel):
    return [kernel(*inputs, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.out_ptr(0),
        np.uint32(M), np.uint32(K),
    ])]
