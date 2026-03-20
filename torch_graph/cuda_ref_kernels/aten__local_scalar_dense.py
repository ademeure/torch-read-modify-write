"""Reference CUDA kernel for aten._local_scalar_dense — GPU to CPU scalar."""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_local_scalar_dense(const float *in0, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out0[i] = in0[i];
}
"""

def init_once():
    x = torch.tensor([3.14], device="cuda")
    return {"kernel_source": KERNEL_SRC, "inputs": [x], "expected": [x]}

def run(inputs, kernel):
    return [kernel(*inputs)]
