"""Reference CUDA kernel for aten.rand — pseudo-random uniform [0,1)."""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_rand(float *out0, unsigned int n, unsigned int seed) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // Simple LCG PRNG (not cryptographic, just deterministic reference)
    unsigned int s = (i + 1u) * 1103515245u + seed * 12345u;
    s = s * 1103515245u + 12345u;
    out0[i] = (float)(s >> 8) / 16777216.0f;  // [0, 1)
}
"""

def init_once():
    n = 1024
    # Can't match PyTorch's RNG exactly — just verify output is in [0,1) and non-trivial
    return {"kernel_source": KERNEL_SRC, "inputs": [],
            "expected": [torch.rand(n, device="cuda")],
            "outputs": ["float32;n=%d" % n], "grid": ((n + 255) // 256,)}

def run(inputs, kernel):
    return [kernel(params=[kernel.out_ptr(0), np.uint32(1024), np.uint32(42)])]
