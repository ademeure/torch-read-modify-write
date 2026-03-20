"""Reference CUDA kernel for aten.randn — Box-Muller pseudo-random normal."""
import torch, numpy as np
KERNEL_SRC = r"""
extern "C" __global__ void aten_randn_kernel(float *out0, unsigned int n, unsigned int seed) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int s1 = (2*i + 1u) * 1103515245u + seed * 12345u;
    s1 = s1 * 1103515245u + 12345u;
    unsigned int s2 = (2*i + 2u) * 1103515245u + seed * 12345u;
    s2 = s2 * 1103515245u + 12345u;
    float u1 = ((float)(s1 >> 8) + 1.0f) / 16777217.0f;
    float u2 = (float)(s2 >> 8) / 16777216.0f;
    out0[i] = sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}
"""
def init_once():
    return {"kernel_source": KERNEL_SRC, "inputs": [], "expected": "skip",
            "outputs": ["float32;n=1024"], "grid": (4,)}
def run(inputs, kernel):
    return [kernel(params=[kernel.out_ptr(0), np.uint32(1024), np.uint32(42)])]
