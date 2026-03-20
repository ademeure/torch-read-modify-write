"""Reference CUDA kernel for aten.slice — copy sub-range along dim 0.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_slice.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_slice_2d(
    const float *input, float *output,
    unsigned int in_rows, unsigned int cols,
    unsigned int start, unsigned int step, unsigned int out_rows
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_rows * cols) return;
    unsigned int r = idx / cols;
    unsigned int c = idx % cols;
    unsigned int src_r = start + r * step;
    output[idx] = input[src_r * cols + c];
}
"""

def init_once():
    x = torch.randn(32, 64, device="cuda")
    out_rows = 16
    total = out_rows * 64
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x],
        "expected": [torch.ops.aten.slice.Tensor(x, 0, 4, 20).contiguous().flatten()],
        "outputs": ["float32;n=%d" % total],
        "grid": ((total + 255) // 256,),
    }

def run(inputs, kernel):
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(32), np.uint32(64),
        np.uint32(4), np.uint32(1), np.uint32(16),
    ])]
