"""Reference CUDA kernel for aten.ones — create one-filled tensor.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_ones.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_fill_one(float *output, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = 1.0f;
}
"""

def init_once():
    return {
        "kernel_source": KERNEL_SRC, "inputs": [],
        "expected": [torch.ones(32, 64, device='cuda').flatten()],
        "outputs": ["float32;n=%d" % (32 * 64)],
        "grid": (((32 * 64) + 255) // 256,),
    }

def run(inputs, kernel):
    return [kernel(params=[
        kernel.out_ptr(0), np.uint32(32 * 64),
    ])]
