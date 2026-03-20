"""Reference CUDA kernel for aten.native_group_norm_backward."""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_group_norm_bwd(
    const float *grad_out, const float *input, const float *mean, const float *rstd,
    const float *weight, float *grad_input,
    unsigned int N, unsigned int C, unsigned int HW, unsigned int G
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = N * C * HW;
    if (idx >= total) return;
    unsigned int hw = idx % HW;
    unsigned int c = (idx / HW) % C;
    unsigned int n = idx / (C * HW);
    unsigned int g = c / (C / G);
    unsigned int ng = n * G + g;
    grad_input[idx] = grad_out[idx] * weight[c] * rstd[ng];
}
"""

NN, CC, HW, GG = 2, 8, 16, 4

def init_once():
    x = torch.randn(NN, CC, 4, 4, device="cuda", requires_grad=True)
    w = torch.randn(CC, device="cuda", requires_grad=True)
    b = torch.randn(CC, device="cuda")
    out, mean, rstd = torch.ops.aten.native_group_norm.default(x, w, b, NN, CC, HW, GG, 1e-5)
    grad = torch.randn_like(out)
    result = torch.ops.aten.native_group_norm_backward.default(grad, x, mean, rstd, w, NN, CC, HW, GG, [True, True, True])
    total = x.numel()
    return {"kernel_source": KERNEL_SRC, "inputs": [grad, x.detach(), mean, rstd, w],
            "expected": [result[0].flatten()],
            "outputs": ["float32;n=%d" % total], "grid": ((total + 255) // 256,), "atol": 2.0}

def run(inputs, kernel):
    total = inputs[0].numel()
    return [kernel(*inputs, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.in_ptr(2), kernel.in_ptr(3), kernel.in_ptr(4),
        kernel.out_ptr(0), np.uint32(NN), np.uint32(CC), np.uint32(HW), np.uint32(GG),
    ])]
