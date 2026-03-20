"""Reference CUDA kernel for aten.prod.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_prod.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_prod(
    const float *input, float *output, unsigned int rows, unsigned int cols
) {
    __shared__ float sdata[256];
    unsigned int row = blockIdx.x, tid = threadIdx.x;
    if (row >= rows) return;
    const float *ri = input + row * cols;
    float v = 1.0f;
    for (unsigned int j = tid; j < cols; j += blockDim.x) {
        v *= ri[j];
    }
    sdata[tid] = v; __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) { sdata[tid] *= sdata[tid + s]; } __syncthreads();
    }
    if (tid == 0) output[row] = sdata[0];
}
"""

def init_once():
    x = torch.rand(8, 16, device='cuda') + 0.5
    cols = x.size(-1)
    rows = x.numel() // cols
    return {
        "kernel_source": KERNEL_SRC,
        "inputs": [x.reshape(rows, cols).contiguous()],
        "expected": [torch.ops.aten.prod.dim_int(x, -1)],
        "outputs": ["float32;n=%d" % rows],
        "grid": (rows,),
        "block": (256,),
        "atol": 0.01,
    }

def run(inputs, kernel):
    x = inputs[0]
    rows, cols = x.shape
    return [kernel(x, params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(rows), np.uint32(cols),
    ])]
