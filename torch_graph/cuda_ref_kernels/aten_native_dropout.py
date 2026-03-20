"""Reference CUDA kernel for aten.native_dropout — deterministic dropout with mask.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_native_dropout.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_dropout_kernel(
    const float *input, const float *mask, float *output,
    float scale, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = input[i] * mask[i] * scale;
}
"""

def init_once():
    x = torch.randn(1024, device="cuda")
    mask = (torch.rand(1024, device="cuda") > 0.5).float()
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x, mask],
        "expected": [x * mask * 2.0],
    }

def run(inputs, kernel):
    x, mask = inputs
    n = x.numel()
    return [kernel(x, mask, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.out_ptr(0),
        np.float32(2.0), np.uint32(n),
    ])]
