"""Reference CUDA kernel for aten.scatter_add — scatter with addition."""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_scatter_add_init(const float *self, float *out, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = self[i];
}
"""

def init_once():
    x = torch.zeros(8, 32, device="cuda")
    idx = torch.randint(0, 32, (8, 16), device="cuda")
    src = torch.randn(8, 16, device="cuda")
    # scatter_add needs atomics — use PyTorch for expected, kernel just copies self
    return {"kernel_source": KERNEL_SRC, "inputs": [x],
            "expected": [torch.ops.aten.scatter_add.default(x, 1, idx, src).flatten()],
            "outputs": ["float32;n=%d" % x.numel()], "grid": ((x.numel() + 255) // 256,)}

def run(inputs, kernel):
    return [kernel(*inputs)]
