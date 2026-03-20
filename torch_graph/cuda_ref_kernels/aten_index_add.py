"""Reference CUDA kernel for aten.index_add — add source into self at indices.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_index_add.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_index_add_kernel(
    const float *self, const long *index, const float *source, float *out,
    unsigned int rows, unsigned int cols, unsigned int n_idx
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = rows * cols;
    if (idx < total) out[idx] = self[idx];
    __syncthreads();
    unsigned int total_src = n_idx * cols;
    if (idx < total_src) {
        unsigned int r = idx / cols, c = idx % cols;
        long dst_r = index[r];
        atomicAdd(&out[dst_r * cols + c], source[idx]);
    }
}
"""

def init_once():
    x = torch.zeros(32, 64, device="cuda")
    idx = torch.tensor([0, 5, 10, 15, 20], device="cuda")
    src = torch.randn(5, 64, device="cuda")
    total = 32 * 64
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x, idx, src],
        "expected": [torch.ops.aten.index_add.default(x, 0, idx, src).flatten()],
        "outputs": ["float32;n=%d" % total], "atol": 1e-5,
        "grid": ((total + 255) // 256,),
    }

def run(inputs, kernel):
    x, idx, src = inputs
    rows, cols = x.shape
    n_idx = idx.numel()
    return [kernel(x, idx, src, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.in_ptr(2), kernel.out_ptr(0),
        np.uint32(rows), np.uint32(cols), np.uint32(n_idx),
    ])]
