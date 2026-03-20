"""Reference CUDA kernel for aten.scalar_tensor — create single-element tensor."""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_scalar_tensor(float *out0, float value) {
    if (threadIdx.x == 0 && blockIdx.x == 0) out0[0] = value;
}
"""

def init_once():
    return {"kernel_source": KERNEL_SRC, "inputs": [],
            "expected": [torch.tensor(3.14, device="cuda")],
            "outputs": ["float32;n=1"], "grid": (1,), "block": (1,)}

def run(inputs, kernel):
    return [kernel(params=[kernel.out_ptr(0), np.float32(3.14)])]
