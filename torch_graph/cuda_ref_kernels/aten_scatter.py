"""Reference CUDA kernel for aten.scatter — scatter values into tensor by index.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_scatter.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_scatter_2d(
    const float *input, const long *index, const float *src, float *output,
    unsigned int rows, unsigned int in_cols, unsigned int src_cols
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_in = rows * in_cols;
    if (idx < total_in) output[idx] = input[idx];
    __syncthreads();
    unsigned int total_src = rows * src_cols;
    if (idx < total_src) {
        unsigned int r = idx / src_cols, c = idx % src_cols;
        long dst_c = index[idx];
        output[r * in_cols + dst_c] = src[idx];
    }
}
"""

def init_once():
    x = torch.zeros(8, 32, device="cuda")
    idx = torch.randint(0, 32, (8, 16), device="cuda")
    src = torch.randn(8, 16, device="cuda")
    total = 8 * 32
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x, idx, src],
        "expected": [torch.ops.aten.scatter.src(x, 1, idx, src).flatten()],
        "outputs": ["float32;n=%d" % total],
        "grid": ((total + 255) // 256,),
    }

def run(inputs, kernel):
    x, idx, src = inputs
    rows, in_cols = x.shape
    src_cols = idx.shape[1]
    return [kernel(x, idx, src, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.in_ptr(2), kernel.out_ptr(0),
        np.uint32(rows), np.uint32(in_cols), np.uint32(src_cols),
    ])]
