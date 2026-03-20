"""Reference CUDA kernel for aten.masked_fill — fill where mask is True."""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_masked_fill(
    const float *input, const float *mask, float *out, float value, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (mask[i] != 0.0f) ? value : input[i];
}
"""

def init_once():
    x = torch.randn(1024, device='cuda')
    mask = (torch.randn(1024, device='cuda') > 0).float()
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x, mask],
        "expected": [torch.ops.aten.masked_fill.Scalar(x, mask.bool(), -1e9)],
    }

def run(inputs, kernel):
    n = inputs[0].numel()
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.out_ptr(0),
        np.float32(-1e9), np.uint32(n),
    ])]
