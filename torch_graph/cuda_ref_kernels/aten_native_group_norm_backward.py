"""Reference CUDA kernel for aten.native_group_norm_backward."""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_group_norm_bwd(
    const float *grad_out, const float *input, const float *mean, const float *rstd,
    const float *weight, float *grad_input,
    unsigned int N, unsigned int C, unsigned int HW, unsigned int G
) {
    // Full 3-term backward per group:
    // dx = (rstd/M) * (M*dy*w - sum_group(dy*w) - x_hat*sum_group(dy*w*x_hat))
    // M = cpg * HW (elements per group)
    unsigned int ng = blockIdx.x;  // one block per (n, g) pair
    if (ng >= N * G) return;
    unsigned int n = ng / G, g = ng % G;
    unsigned int cpg = C / G;
    unsigned int M = cpg * HW;
    float m = mean[ng];
    float rs = rstd[ng];
    float inv_M = 1.0f / (float)M;
    unsigned int base = n * C * HW + g * cpg * HW;
    // Pass 1: sum(dy*w) and sum(dy*w*x_hat) over all elements in this group
    float sum_dxhat = 0.0f, sum_dxhat_xhat = 0.0f;
    for (unsigned int ci = 0; ci < cpg; ci++) {
        unsigned int c = g * cpg + ci;
        float wc = weight[c];
        for (unsigned int hw = 0; hw < HW; hw++) {
            unsigned int idx = base + ci * HW + hw;
            float xhat = (input[idx] - m) * rs;
            float dxh = grad_out[idx] * wc;
            sum_dxhat += dxh;
            sum_dxhat_xhat += dxh * xhat;
        }
    }
    // Pass 2: compute grad_input
    for (unsigned int ci = 0; ci < cpg; ci++) {
        unsigned int c = g * cpg + ci;
        float wc = weight[c];
        for (unsigned int hw = 0; hw < HW; hw++) {
            unsigned int idx = base + ci * HW + hw;
            float xhat = (input[idx] - m) * rs;
            float dxh = grad_out[idx] * wc;
            grad_input[idx] = rs * inv_M * ((float)M * dxh - sum_dxhat - xhat * sum_dxhat_xhat);
        }
    }
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
            "outputs": ["float32;n=%d" % total], "grid": (NN * GG,), "block": (1,), "atol": 1e-2}

def run(inputs, kernel):
    total = inputs[0].numel()
    return [kernel(*inputs, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.in_ptr(2), kernel.in_ptr(3), kernel.in_ptr(4),
        kernel.out_ptr(0), np.uint32(NN), np.uint32(CC), np.uint32(HW), np.uint32(GG),
    ])]
