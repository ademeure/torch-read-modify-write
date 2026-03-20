"""Reference CUDA kernel for aten.reflection_pad2d."""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_reflection_pad2d(
    const float *input, float *output,
    unsigned int N, unsigned int C, unsigned int H, unsigned int W,
    unsigned int outH, unsigned int outW,
    unsigned int padT, unsigned int padL
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = N * C * outH * outW;
    if (idx >= total) return;
    unsigned int ow = idx % outW;
    unsigned int oh = (idx / outW) % outH;
    unsigned int c = (idx / (outW * outH)) % C;
    unsigned int n = idx / (outW * outH * C);
    // Reflect coordinates
    int ih = (int)oh - (int)padT;
    int iw = (int)ow - (int)padL;
    if (ih < 0) ih = -ih;
    if (iw < 0) iw = -iw;
    if (ih >= (int)H) ih = 2*(int)H - ih - 2;
    if (iw >= (int)W) iw = 2*(int)W - iw - 2;
    output[idx] = input[n*C*H*W + c*H*W + ih*W + iw];
}
"""

def init_once():
    x = torch.randn(1, 4, 8, 8, device="cuda")
    padL, padR, padT, padB = 2, 2, 2, 2
    outH, outW = 8 + padT + padB, 8 + padL + padR
    total = 1 * 4 * outH * outW
    return {"kernel_source": KERNEL_SRC, "inputs": [x.contiguous()],
            "expected": [torch.ops.aten.reflection_pad2d.default(x, [padL, padR, padT, padB]).flatten()],
            "outputs": ["float32;n=%d" % total], "grid": ((total + 255) // 256,)}

def run(inputs, kernel):
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(1), np.uint32(4), np.uint32(8), np.uint32(8),
        np.uint32(12), np.uint32(12), np.uint32(2), np.uint32(2),
    ])]
