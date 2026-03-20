"""Reference CUDA kernel for aten.cat — concatenate tensors along dim 0.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_cat.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_cat_dim0(
    const float *a, const float *b, float *out,
    unsigned int a_rows, unsigned int b_rows, unsigned int cols
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = (a_rows + b_rows) * cols;
    if (idx >= total) return;
    unsigned int r = idx / cols, c = idx % cols;
    if (r < a_rows)
        out[idx] = a[r * cols + c];
    else
        out[idx] = b[(r - a_rows) * cols + c];
}
"""

def init_once():
    a = torch.randn(8, 32, device="cuda")
    b = torch.randn(16, 32, device="cuda")
    total = (8 + 16) * 32
    return {
        "kernel_source": KERNEL_SRC, "inputs": [a, b],
        "expected": [torch.ops.aten.cat.default([a, b], 0).flatten()],
        "outputs": ["float32;n=%d" % total],
        "grid": ((total + 255) // 256,),
    }

def run(inputs, kernel):
    a, b = inputs
    return [kernel(a, b, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.out_ptr(0),
        np.uint32(a.shape[0]), np.uint32(b.shape[0]), np.uint32(a.shape[1]),
    ])]
