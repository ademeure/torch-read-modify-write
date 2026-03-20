"""Reference CUDA kernel for aten.max_pool2d_with_indices_backward."""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_maxpool2d_bwd(
    const float *grad_output, const long *indices, float *grad_input,
    unsigned int total_in, unsigned int C, unsigned int H, unsigned int W,
    unsigned int outH, unsigned int outW
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_in) return;
    grad_input[idx] = 0.0f;
}
extern "C" __global__ void aten_maxpool2d_scatter(
    const float *grad_output, const long *indices, float *grad_input,
    unsigned int total_out, unsigned int C, unsigned int H, unsigned int W, unsigned int outH, unsigned int outW
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_out) return;
    unsigned int ow = idx % outW, oh = (idx / outW) % outH;
    unsigned int c = (idx / (outW * outH)) % C;
    unsigned int n = idx / (outW * outH * C);
    long flat_idx = indices[idx];
    unsigned int ih = flat_idx / W, iw = flat_idx % W;
    atomicAdd(&grad_input[n*C*H*W + c*H*W + ih*W + iw], grad_output[idx]);
}
"""

def init_once():
    x = torch.randn(1, 4, 8, 8, device="cuda")
    out, indices = torch.ops.aten.max_pool2d_with_indices.default(x, [2,2], [2,2])
    grad = torch.randn_like(out)
    result = torch.ops.aten.max_pool2d_with_indices_backward.default(grad, x, [2,2], [2,2], [0,0], [1,1], False, indices)
    total = x.numel()
    return {"kernel_source": KERNEL_SRC, "inputs": [grad.contiguous(), indices.contiguous()],
            "expected": [result.flatten()],
            "outputs": ["float32;n=%d" % total], "grid": ((total + 255) // 256,), "atol": 1e-5}

def run(inputs, kernel):
    return [kernel(*inputs)]
