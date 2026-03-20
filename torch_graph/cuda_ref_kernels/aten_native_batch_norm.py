"""Reference CUDA kernel for aten.native_batch_norm — per-channel normalization.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_native_batch_norm.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_batch_norm(
    const float *input, const float *weight, const float *bias,
    const float *running_mean, const float *running_var,
    float *output, unsigned int N, unsigned int C, unsigned int HW, float eps
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = N * C * HW;
    if (idx < total) {
        unsigned int c = (idx / HW) % C;
        float mean = running_mean[c];
        float var = running_var[c];
        float x = input[idx];
        float normed = (x - mean) * rsqrtf(var + eps);
        output[idx] = normed * weight[c] + bias[c];
    }
}
"""

NN, CC, HH, WW = 2, 8, 4, 4

def init_once():
    x = torch.randn(NN, CC, HH, WW, device="cuda")
    w = torch.randn(CC, device="cuda")
    b = torch.randn(CC, device="cuda")
    rm = torch.randn(CC, device="cuda")
    rv = torch.rand(CC, device="cuda") + 0.1
    total = NN * CC * HH * WW
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x.contiguous(), w, b, rm, rv],
        "expected": [torch.ops.aten.native_batch_norm.default(x, w, b, rm, rv, False, 0.1, 1e-5)[0].flatten()],
        "outputs": ["float32;n=%d" % total], "atol": 1e-4,
        "grid": ((total + 255) // 256,),
    }

def run(inputs, kernel):
    total = inputs[0].numel()
    HW = HH * WW
    return [kernel(*inputs, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.in_ptr(2),
        kernel.in_ptr(3), kernel.in_ptr(4), kernel.out_ptr(0),
        np.uint32(NN), np.uint32(CC), np.uint32(HW), np.float32(1e-5),
    ])]
