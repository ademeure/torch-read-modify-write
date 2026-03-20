"""Reference CUDA kernel for aten.stack — stack tensors along new dim 0.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_stack.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_stack_2(
    const float *a, const float *b, float *out, unsigned int L
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < L) {
        out[i] = a[i];
        out[L + i] = b[i];
    }
}
"""

def init_once():
    a = torch.randn(64, device="cuda")
    b = torch.randn(64, device="cuda")
    return {
        "kernel_source": KERNEL_SRC, "inputs": [a, b],
        "expected": [torch.ops.aten.stack.default([a, b], 0).flatten()],
        "outputs": ["float32;n=128"],
        "grid": (1,),
    }

def run(inputs, kernel):
    a, b = inputs
    L = a.numel()
    return [kernel(a, b, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.out_ptr(0),
        np.uint32(L),
    ])]
