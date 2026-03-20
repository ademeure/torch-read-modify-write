"""Reference CUDA kernel for aten.select — select single index along dim 0.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_select.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_select_copy(
    const float *input, float *output, unsigned int cols, unsigned int index
) {
    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < cols) output[c] = input[index * cols + c];
}
"""

def init_once():
    x = torch.randn(32, 64, device="cuda")
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x],
        "expected": [torch.ops.aten.select.int(x, 0, 5).contiguous().flatten()],
        "outputs": ["float32;n=64"],
        "grid": (1,),
    }

def run(inputs, kernel):
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(64), np.uint32(5),
    ])]
