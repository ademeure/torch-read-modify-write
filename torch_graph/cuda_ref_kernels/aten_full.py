"""Reference CUDA kernel for aten.full — create tensor filled with a value.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_full.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_fill_val(float *output, float value, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = value;
}
"""

def init_once():
    return {
        "kernel_source": KERNEL_SRC, "inputs": [],
        "expected": [torch.full((32, 64), 3.14, device='cuda').flatten()],
        "outputs": ["float32;n=%d" % (32 * 64)],
        "grid": (((32 * 64) + 255) // 256,),
    }

def run(inputs, kernel):
    return [kernel(params=[
        kernel.out_ptr(0), np.float32(3.14), np.uint32(32 * 64),
    ])]
