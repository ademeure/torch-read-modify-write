"""Reference CUDA kernel for aten.full_like — fill tensor shape with value."""
import torch

KERNEL_SRC = r"""
extern "C" __global__ void aten_full_like(float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out0[i] = 3.14f;
}
"""

def init_once():
    x = torch.randn(1024, device="cuda")
    return {"kernel_source": KERNEL_SRC, "inputs": [x],
            "expected": [torch.full_like(x, 3.14)]}

def run(inputs, kernel):
    return [kernel(*inputs)]
