"""Reference CUDA kernel for aten.expand — broadcast copy to larger shape.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_expand.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_expand_2d(
    const float *input, float *output,
    unsigned int in_rows, unsigned int in_cols,
    unsigned int out_rows, unsigned int out_cols
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_rows * out_cols) return;
    unsigned int r = idx / out_cols;
    unsigned int c = idx % out_cols;
    unsigned int ir = (in_rows == 1) ? 0 : r;
    unsigned int ic = (in_cols == 1) ? 0 : c;
    output[idx] = input[ir * in_cols + ic];
}
"""

def init_once():
    x = torch.randn(1, 64, device="cuda")
    total = 32 * 64
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x],
        "expected": [torch.ops.aten.expand.default(x, [32, 64]).contiguous().flatten()],
        "outputs": ["float32;n=%d" % total],
        "grid": ((total + 255) // 256,),
    }

def run(inputs, kernel):
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(1), np.uint32(64),
        np.uint32(32), np.uint32(64),
    ])]
