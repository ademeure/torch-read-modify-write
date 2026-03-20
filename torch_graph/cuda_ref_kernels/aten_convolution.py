"""Reference CUDA kernel for aten.convolution — naive conv2d.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_convolution.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_conv2d_kernel(
    const float *input, const float *weight, const float *bias, float *output,
    unsigned int N, unsigned int C_in, unsigned int H, unsigned int W,
    unsigned int C_out, unsigned int kH, unsigned int kW,
    unsigned int padH, unsigned int padW, unsigned int strideH, unsigned int strideW,
    unsigned int outH, unsigned int outW
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = N * C_out * outH * outW;
    if (idx >= total) return;
    unsigned int ow = idx % outW;
    unsigned int oh = (idx / outW) % outH;
    unsigned int oc = (idx / (outW * outH)) % C_out;
    unsigned int n = idx / (outW * outH * C_out);
    float sum = bias[oc];
    for (unsigned int ic = 0; ic < C_in; ic++) {
        for (unsigned int kh = 0; kh < kH; kh++) {
            for (unsigned int kw = 0; kw < kW; kw++) {
                int ih = (int)(oh * strideH + kh) - (int)padH;
                int iw = (int)(ow * strideW + kw) - (int)padW;
                if (ih >= 0 && ih < (int)H && iw >= 0 && iw < (int)W) {
                    sum += input[n*C_in*H*W + ic*H*W + ih*W + iw]
                         * weight[oc*C_in*kH*kW + ic*kH*kW + kh*kW + kw];
                }
            }
        }
    }
    output[idx] = sum;
}
"""

NN, C_IN, H, W, C_OUT, KH, KW = 1, 3, 8, 8, 16, 3, 3
PAD, STRIDE = 1, 1
OUT_H = (H + 2 * PAD - KH) // STRIDE + 1
OUT_W = (W + 2 * PAD - KW) // STRIDE + 1

def init_once():
    x = torch.randn(NN, C_IN, H, W, device="cuda")
    w = torch.randn(C_OUT, C_IN, KH, KW, device="cuda")
    b = torch.randn(C_OUT, device="cuda")
    total = NN * C_OUT * OUT_H * OUT_W
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x.contiguous(), w.contiguous(), b],
        "expected": [torch.ops.aten.convolution.default(x, w, b, [STRIDE,STRIDE], [PAD,PAD], [1,1], False, [0,0], 1).flatten()],
        "outputs": ["float32;n=%d" % total],
        "grid": ((total + 255) // 256,), "atol": 1e-3,
    }

def run(inputs, kernel):
    total = NN * C_OUT * OUT_H * OUT_W
    return [kernel(*inputs, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.in_ptr(2), kernel.out_ptr(0),
        np.uint32(NN), np.uint32(C_IN), np.uint32(H), np.uint32(W),
        np.uint32(C_OUT), np.uint32(KH), np.uint32(KW),
        np.uint32(PAD), np.uint32(PAD), np.uint32(STRIDE), np.uint32(STRIDE),
        np.uint32(OUT_H), np.uint32(OUT_W),
    ])]
