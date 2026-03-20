"""Reference CUDA kernel for aten.linear — y = x @ weight.T + bias.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_linear.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_linear(
    const float *x, const float *w, const float *bias, float *out,
    unsigned int M, unsigned int K, unsigned int N
) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = bias[col];
        for (unsigned int k = 0; k < K; k++)
            sum += x[row * K + k] * w[col * K + k];
        out[row * N + col] = sum;
    }
}
"""

M, K, N = 32, 64, 48

def init_once():
    x = torch.randn(M, K, device="cuda")
    w = torch.randn(N, K, device="cuda")
    b = torch.randn(N, device="cuda")
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x, w, b],
        "expected": [torch.ops.aten.linear.default(x, w, b).flatten()],
        "outputs": ["float32;n=%d" % (M * N)],
        "grid": ((N + 15) // 16, (M + 15) // 16),
        "block": (16, 16), "atol": 1e-3,
    }

def run(inputs, kernel):
    return [kernel(*inputs, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.in_ptr(2), kernel.out_ptr(0),
        np.uint32(M), np.uint32(K), np.uint32(N),
    ])]
