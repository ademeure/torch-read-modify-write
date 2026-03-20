"""Reference CUDA kernel for aten.index_select — select rows by index.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_index_select.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_index_select_kernel(
    const float *input, const long *index, float *output,
    unsigned int n_idx, unsigned int cols
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_idx * cols) return;
    unsigned int r = idx / cols, c = idx % cols;
    long src_r = index[r];
    output[idx] = input[src_r * cols + c];
}
"""

def init_once():
    x = torch.randn(32, 64, device="cuda")
    idx = torch.tensor([0, 5, 10, 15, 31], device="cuda")
    total = 5 * 64
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x, idx],
        "expected": [torch.ops.aten.index_select.default(x, 0, idx).flatten()],
        "outputs": ["float32;n=%d" % total],
        "grid": ((total + 255) // 256,),
    }

def run(inputs, kernel):
    x, idx = inputs
    n_idx = idx.numel()
    cols = x.shape[1]
    return [kernel(x, idx, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.out_ptr(0),
        np.uint32(n_idx), np.uint32(cols),
    ])]
