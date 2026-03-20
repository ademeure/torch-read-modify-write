"""Reference CUDA kernel for aten.convolution_backward — naive grad_input."""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_conv_bwd_input(
    const float *grad_output, const float *weight, float *grad_input,
    unsigned int N, unsigned int C_in, unsigned int H, unsigned int W,
    unsigned int C_out, unsigned int kH, unsigned int kW,
    unsigned int padH, unsigned int padW, unsigned int strideH, unsigned int strideW,
    unsigned int outH, unsigned int outW
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = N * C_in * H * W;
    if (idx >= total) return;
    unsigned int iw = idx % W, ih = (idx / W) % H;
    unsigned int ic = (idx / (W * H)) % C_in;
    unsigned int n = idx / (W * H * C_in);
    float sum = 0.0f;
    for (unsigned int oc = 0; oc < C_out; oc++) {
        for (unsigned int kh = 0; kh < kH; kh++) {
            for (unsigned int kw = 0; kw < kW; kw++) {
                int oh = ((int)ih + (int)padH - (int)kh);
                int ow = ((int)iw + (int)padW - (int)kw);
                if (oh % (int)strideH == 0 && ow % (int)strideW == 0) {
                    oh /= (int)strideH; ow /= (int)strideW;
                    if (oh >= 0 && oh < (int)outH && ow >= 0 && ow < (int)outW)
                        sum += grad_output[n*C_out*outH*outW + oc*outH*outW + oh*outW + ow]
                             * weight[oc*C_in*kH*kW + ic*kH*kW + kh*kW + kw];
                }
            }
        }
    }
    grad_input[idx] = sum;
}
"""

NN, C_IN, H, W, C_OUT, KH, KW, PAD, STRIDE = 1, 3, 8, 8, 4, 3, 3, 1, 1
OUT_H = (H + 2*PAD - KH) // STRIDE + 1
OUT_W = (W + 2*PAD - KW) // STRIDE + 1

def init_once():
    grad_out = torch.randn(NN, C_OUT, OUT_H, OUT_W, device="cuda")
    weight = torch.randn(C_OUT, C_IN, KH, KW, device="cuda")
    total = NN * C_IN * H * W
    x = torch.randn(NN, C_IN, H, W, device="cuda", requires_grad=True)
    result = torch.ops.aten.convolution_backward.default(
        grad_out, x, weight, [0], [STRIDE,STRIDE], [PAD,PAD], [1,1], False, [0,0], 1, [True, True, True])
    return {"kernel_source": KERNEL_SRC, "inputs": [grad_out.contiguous(), weight.contiguous()],
            "expected": [result[0].flatten()],
            "outputs": ["float32;n=%d" % total], "grid": ((total + 255) // 256,), "atol": 1e-3}

def run(inputs, kernel):
    total = NN * C_IN * H * W
    return [kernel(*inputs, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.out_ptr(0),
        np.uint32(NN), np.uint32(C_IN), np.uint32(H), np.uint32(W),
        np.uint32(C_OUT), np.uint32(KH), np.uint32(KW),
        np.uint32(PAD), np.uint32(PAD), np.uint32(STRIDE), np.uint32(STRIDE),
        np.uint32(OUT_H), np.uint32(OUT_W),
    ])]
