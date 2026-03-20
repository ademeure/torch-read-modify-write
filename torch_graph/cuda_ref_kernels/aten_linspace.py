"""Reference CUDA kernel for aten.linspace — evenly spaced values.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_linspace.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_linspace_kernel(
    float *output, float start, float step, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = start + i * step;
}
"""

def init_once():
    return {
        "kernel_source": KERNEL_SRC, "inputs": [],
        "expected": [torch.linspace(0, 1, 100, device='cuda')],
        "outputs": ["float32;n=100"], "grid": ((100 + 255) // 256,), "atol": 1e-5,
    }

def run(inputs, kernel):
    step = 1.0 / 99.0
    return [kernel(params=[
        kernel.out_ptr(0), np.float32(0.0), np.float32(step), np.uint32(100),
    ])]
