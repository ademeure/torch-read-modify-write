"""Reference CUDA kernel for aten.native_layer_norm_backward."""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_layer_norm_bwd(
    const float *grad_out, const float *input, const float *mean, const float *rstd,
    const float *weight, float *grad_input,
    unsigned int rows, unsigned int cols
) {
    // Full 3-term backward: dx = (rstd/N) * (N*dy*w - sum(dy*w) - x_hat*sum(dy*w*x_hat))
    unsigned int row = blockIdx.x;
    if (row >= rows) return;
    float m = mean[row];
    float rs = rstd[row];
    float inv_N = 1.0f / (float)cols;
    const float *go = grad_out + row * cols;
    const float *x = input + row * cols;
    float *dx = grad_input + row * cols;
    // Pass 1: compute sum(dy*w) and sum(dy*w*x_hat)
    float sum_dxhat = 0.0f, sum_dxhat_xhat = 0.0f;
    for (unsigned int j = 0; j < cols; j++) {
        float xhat = (x[j] - m) * rs;
        float dxh = go[j] * weight[j];
        sum_dxhat += dxh;
        sum_dxhat_xhat += dxh * xhat;
    }
    // Pass 2: compute grad_input
    for (unsigned int j = 0; j < cols; j++) {
        float xhat = (x[j] - m) * rs;
        float dxh = go[j] * weight[j];
        dx[j] = rs * inv_N * ((float)cols * dxh - sum_dxhat - xhat * sum_dxhat_xhat);
    }
}
"""

def init_once():
    x = torch.randn(8, 64, device="cuda", requires_grad=True)
    w = torch.randn(64, device="cuda", requires_grad=True)
    b = torch.randn(64, device="cuda")
    out, mean, rstd = torch.ops.aten.native_layer_norm.default(x, [64], w, b, 1e-5)
    grad = torch.randn_like(out)
    result = torch.ops.aten.native_layer_norm_backward.default(grad, x, [64], mean, rstd, w, b, [True, True, True])
    total = x.numel()
    return {"kernel_source": KERNEL_SRC, "inputs": [grad, x.detach(), mean, rstd, w],
            "expected": [result[0].flatten()],
            "outputs": ["float32;n=%d" % total], "grid": (8,), "block": (1,), "atol": 1e-2}

def run(inputs, kernel):
    total = inputs[0].numel()
    rows, cols = inputs[0].size(0), inputs[0].size(1)
    return [kernel(*inputs, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.in_ptr(2), kernel.in_ptr(3), kernel.in_ptr(4),
        kernel.out_ptr(0), np.uint32(rows), np.uint32(cols),
    ])]
