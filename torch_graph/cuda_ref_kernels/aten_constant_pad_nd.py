"""Reference CUDA kernel for aten.constant_pad_nd — 2D constant padding.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_constant_pad_nd.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_constant_pad_2d(
    const float *input, float *output,
    unsigned int H, unsigned int W, unsigned int outH, unsigned int outW,
    unsigned int padTop, unsigned int padLeft, float value, unsigned int total
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    unsigned int ow = idx % outW;
    unsigned int oh = (idx / outW) % outH;
    unsigned int n = idx / (outH * outW);
    int ih = (int)oh - (int)padTop;
    int iw = (int)ow - (int)padLeft;
    if (ih >= 0 && ih < (int)H && iw >= 0 && iw < (int)W)
        output[idx] = input[n * H * W + ih * W + iw];
    else
        output[idx] = value;
}
"""

NN, H, W = 2, 8, 8
PAD = 1
OUT_H = H + 2 * PAD
OUT_W = W + 2 * PAD

def init_once():
    x = torch.randn(NN, H, W, device="cuda")
    total = NN * OUT_H * OUT_W
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x],
        "expected": [torch.ops.aten.constant_pad_nd.default(x, [PAD, PAD, PAD, PAD], 0.0).flatten()],
        "outputs": ["float32;n=%d" % total],
        "grid": ((total + 255) // 256,),
    }

def run(inputs, kernel):
    total = NN * OUT_H * OUT_W
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(H), np.uint32(W), np.uint32(OUT_H), np.uint32(OUT_W),
        np.uint32(PAD), np.uint32(PAD), np.float32(0.0), np.uint32(total),
    ])]
