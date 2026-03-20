"""Reference CUDA kernel for aten.rand — LCG pseudo-random uniform [0,1)."""
import torch, numpy as np
KERNEL_SRC = r"""
extern "C" __global__ void aten_rand_kernel(float *out0, unsigned int n, unsigned int seed) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int s = (i + 1u) * 1103515245u + seed * 12345u;
    s = s * 1103515245u + 12345u;
    out0[i] = (float)(s >> 8) / 16777216.0f;
}
"""
def init_once():
    return {"kernel_source": KERNEL_SRC, "inputs": [], "expected": "skip",
            "outputs": ["float32;n=1024"], "grid": (4,)}
def run(inputs, kernel):
    return [kernel(params=[kernel.out_ptr(0), np.uint32(1024), np.uint32(42)])]
