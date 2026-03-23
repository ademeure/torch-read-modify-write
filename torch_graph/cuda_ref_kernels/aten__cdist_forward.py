"""Reference CUDA kernel for aten._cdist_forward — pairwise L2 distances."""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_cdist(
    const float *x1, const float *x2, float *out,
    unsigned int B, unsigned int M, unsigned int N, unsigned int D
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = B * M * N;
    if (idx >= total) return;
    unsigned int n = idx % N, m = (idx / N) % M, b = idx / (N * M);
    float sum = 0.0f;
    for (unsigned int d = 0; d < D; d++) {
        float diff = x1[b*M*D + m*D + d] - x2[b*N*D + n*D + d];
        sum += (diff != diff) ? __int_as_float(0x7FC00000) : diff * diff;
    }
    out[idx] = sqrtf(sum);
}
"""

BB, MM, NN, DD = 2, 8, 6, 4

def init_once():
    x1 = torch.randn(BB, MM, DD, device="cuda")
    x2 = torch.randn(BB, NN, DD, device="cuda")
    total = BB * MM * NN
    return {"kernel_source": KERNEL_SRC, "inputs": [x1.contiguous(), x2.contiguous()],
            "expected": [torch.ops.aten._cdist_forward.default(x1, x2, 2.0, None).flatten()],
            "outputs": ["float32;n=%d" % total], "grid": ((total + 255) // 256,), "atol": 1e-4}

def run(inputs, kernel):
    total = BB * MM * NN
    return [kernel(*inputs, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.out_ptr(0),
        np.uint32(BB), np.uint32(MM), np.uint32(NN), np.uint32(DD),
    ])]
