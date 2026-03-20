"""Reference CUDA kernel for aten.var — variance reduction.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_var.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_var_kernel(
    const float *input, float *output, unsigned int rows, unsigned int cols, int correction
) {
    __shared__ float s_sum[256];
    __shared__ float s_sq[256];
    unsigned int row = blockIdx.x, tid = threadIdx.x;
    if (row >= rows) return;
    const float *ri = input + row * cols;
    float ls = 0.0f, lsq = 0.0f;
    for (unsigned int j = tid; j < cols; j += blockDim.x) {
        float v = ri[j]; ls += v; lsq += v * v;
    }
    s_sum[tid] = ls; s_sq[tid] = lsq; __syncthreads();
    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) { s_sum[tid] += s_sum[tid+s]; s_sq[tid] += s_sq[tid+s]; }
        __syncthreads();
    }
    if (tid == 0) {
        float mean = s_sum[0] / (float)cols;
        output[row] = (s_sq[0] / (float)cols - mean * mean) * (float)cols / (float)(cols - correction);
    }
}
"""

ROWS, COLS = 32, 64

def init_once():
    x = torch.randn(ROWS, COLS, device="cuda")
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x],
        "expected": [torch.ops.aten.var.correction(x, [-1])],
        "outputs": ["float32;n=%d" % ROWS],
        "grid": (ROWS,),
        "block": (256,), "atol": 1e-3,
    }

def run(inputs, kernel):
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(ROWS), np.uint32(COLS), np.int32(1),
    ])]
