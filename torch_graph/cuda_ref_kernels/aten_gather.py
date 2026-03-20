"""Reference CUDA kernel for aten.gather — gather along dim by index.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_gather.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_gather_2d(
    const float *input, const long *index, float *output,
    unsigned int rows, unsigned int in_cols, unsigned int out_cols
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * out_cols) return;
    unsigned int r = idx / out_cols, c = idx % out_cols;
    long src_c = index[r * out_cols + c];
    output[idx] = input[r * in_cols + src_c];
}
"""

def init_once():
    x = torch.randn(8, 32, device="cuda")
    idx = torch.randint(0, 32, (8, 16), device="cuda")
    total = 8 * 16
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x, idx],
        "expected": [torch.ops.aten.gather.default(x, 1, idx).flatten()],
        "outputs": ["float32;n=%d" % total],
        "grid": ((total + 255) // 256,),
    }

def run(inputs, kernel):
    x, idx = inputs
    rows, in_cols = x.shape
    out_cols = idx.shape[1]
    return [kernel(x, idx, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.out_ptr(0),
        np.uint32(rows), np.uint32(in_cols), np.uint32(out_cols),
    ])]
