"""Reference CUDA kernel for aten.native_group_norm — group normalization.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_native_group_norm.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_group_norm(
    const float *input, const float *weight, const float *bias,
    float *output, unsigned int N, unsigned int C, unsigned int HW,
    unsigned int G, float eps
) {
    unsigned int ng = blockIdx.x;
    unsigned int n = ng / G, g = ng % G;
    unsigned int tid = threadIdx.x;
    unsigned int CpG = C / G;
    unsigned int group_size = CpG * HW;

    __shared__ float s_sum[256];
    __shared__ float s_sq[256];

    float ls = 0.0f, lsq = 0.0f;
    for (unsigned int i = tid; i < group_size; i += blockDim.x) {
        unsigned int c_local = i / HW;
        unsigned int hw = i % HW;
        unsigned int c = g * CpG + c_local;
        float v = input[n * C * HW + c * HW + hw];
        ls += v; lsq += v * v;
    }
    s_sum[tid] = ls; s_sq[tid] = lsq;
    __syncthreads();
    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) { s_sum[tid] += s_sum[tid+s]; s_sq[tid] += s_sq[tid+s]; }
        __syncthreads();
    }
    float mean = s_sum[0] / (float)group_size;
    float rstd = rsqrtf(s_sq[0] / (float)group_size - mean * mean + eps);

    for (unsigned int i = tid; i < group_size; i += blockDim.x) {
        unsigned int c_local = i / HW;
        unsigned int hw = i % HW;
        unsigned int c = g * CpG + c_local;
        unsigned int idx = n * C * HW + c * HW + hw;
        output[idx] = (input[idx] - mean) * rstd * weight[c] + bias[c];
    }
}
"""

NN, CC, HH, WW, GG = 2, 8, 4, 4, 4

def init_once():
    HW = HH * WW
    x = torch.randn(NN, CC, HH, WW, device="cuda")
    w = torch.randn(CC, device="cuda")
    b = torch.randn(CC, device="cuda")
    total = NN * CC * HW
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x.contiguous(), w, b],
        "expected": [torch.ops.aten.native_group_norm.default(x, w, b, NN, CC, HW, GG, 1e-5)[0].flatten()],
        "outputs": ["float32;n=%d" % total],
        "grid": (NN * GG,),
        "block": (256,), "atol": 1e-4,
    }

def run(inputs, kernel):
    HW = HH * WW
    return [kernel(*inputs, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.in_ptr(2), kernel.out_ptr(0),
        np.uint32(NN), np.uint32(CC), np.uint32(HW),
        np.uint32(GG), np.float32(1e-5),
    ])]
