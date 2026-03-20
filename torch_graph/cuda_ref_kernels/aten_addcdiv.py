"""Reference CUDA kernel for aten.addcdiv — input + value * t1 / t2."""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_addcdiv(
    const float *inp, const float *t1, const float *t2,
    float *out, float value, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = inp[i] + value * t1[i] / t2[i];
}
"""

def init_once():
    inp = torch.randn(1024, device='cuda')
    t1 = torch.randn(1024, device='cuda')
    t2 = torch.randn(1024, device='cuda').abs() + 0.1
    return {
        "kernel_source": KERNEL_SRC, "inputs": [inp, t1, t2],
        "expected": [torch.ops.aten.addcdiv.default(inp, t1, t2, value=0.5)], "atol": 1e-4,
    }

def run(inputs, kernel):
    n = inputs[0].numel()
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.in_ptr(2),
        kernel.out_ptr(0), np.float32(0.5), np.uint32(n),
    ])]
