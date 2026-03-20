"""Reference CUDA kernel for aten.baddbmm — batch add + batch matmul.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_baddbmm.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
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
"""

BATCH, M, K, N = 4, 16, 32, 24

def init_once():
    s = torch.randn(BATCH, M, N, device="cuda")
    A = torch.randn(BATCH, M, K, device="cuda")
    B = torch.randn(BATCH, K, N, device="cuda")
    return {
        "kernel_source": KERNEL_SRC, "inputs": [s, A, B],
        "expected": [torch.ops.aten.baddbmm.default(s, A, B).flatten()],
        "outputs": ["float32;n=%d" % (BATCH * M * N)],
        "grid": ((N + 15) // 16, (M + 15) // 16, BATCH),
        "block": (16, 16), "atol": 1e-3,
    }

def run(inputs, kernel):
    return [kernel(*inputs, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.in_ptr(2), kernel.out_ptr(0),
        np.uint32(BATCH), np.uint32(M), np.uint32(K), np.uint32(N),
        np.float32(1.0), np.float32(1.0),
    ])]
