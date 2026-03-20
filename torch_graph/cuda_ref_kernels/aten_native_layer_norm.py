"""Reference CUDA kernel for aten.native_layer_norm.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_native_layer_norm.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_layer_norm(
    const float *input, const float *weight, const float *bias,
    float *output, unsigned int rows, unsigned int cols, float eps
) {
    __shared__ float s_sum[256];
    __shared__ float s_sq[256];
    unsigned int row = blockIdx.x, tid = threadIdx.x;
    if (row >= rows) return;
    const float *ri = input + row * cols;
    float *ro = output + row * cols;
    float ls = 0.0f, lsq = 0.0f;
    for (unsigned int j = tid; j < cols; j += blockDim.x) {
        float v = ri[j]; ls += v; lsq += v * v;
    }
    s_sum[tid] = ls; s_sq[tid] = lsq; __syncthreads();
    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) { s_sum[tid] += s_sum[tid+s]; s_sq[tid] += s_sq[tid+s]; }
        __syncthreads();
    }
    float mean = s_sum[0] / (float)cols;
    float rstd = rsqrtf(s_sq[0] / (float)cols - mean * mean + eps);
    __syncthreads();
    for (unsigned int j = tid; j < cols; j += blockDim.x)
        ro[j] = (ri[j] - mean) * rstd * weight[j] + bias[j];
}
"""

ROWS, COLS = 8, 64

def init_once():
    x = torch.randn(ROWS, COLS, device="cuda")
    w = torch.randn(COLS, device="cuda")
    b = torch.randn(COLS, device="cuda")
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x, w, b],
        "expected": [torch.ops.aten.native_layer_norm.default(x, [COLS], w, b, 1e-5)[0].flatten()],
        "outputs": ["float32;n=%d" % (ROWS * COLS)],
        "grid": (ROWS,),
        "block": (256,), "atol": 1e-4,
    }

def run(inputs, kernel):
    return [kernel(*inputs, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.in_ptr(2), kernel.out_ptr(0),
        np.uint32(ROWS), np.uint32(COLS), np.float32(1e-5),
    ])]
