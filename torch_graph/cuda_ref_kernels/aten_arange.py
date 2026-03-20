"""Reference CUDA kernel for aten.arange — fill with sequential values.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_arange.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_arange_kernel(
    float *output, float start, float step, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = start + i * step;
}
"""

def init_once():
    return {
        "kernel_source": KERNEL_SRC, "inputs": [],
        "expected": [torch.ops.aten.arange.start_step(0, 100, 1, dtype=torch.float32, device='cuda')],
        "outputs": ["float32;n=100"],
        "grid": (1,),
    }

def run(inputs, kernel):
    return [kernel(params=[
        kernel.out_ptr(0),
        np.float32(0.0), np.float32(1.0), np.uint32(100),
    ])]
