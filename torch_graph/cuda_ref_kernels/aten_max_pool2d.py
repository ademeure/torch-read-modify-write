"""Reference CUDA kernel for aten.max_pool2d — max pooling (values only).
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_max_pool2d.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_max_pool2d_kernel(
    const float *input, float *output,
    unsigned int N, unsigned int C, unsigned int H, unsigned int W,
    unsigned int kH, unsigned int kW, unsigned int strideH, unsigned int strideW,
    unsigned int padH, unsigned int padW, unsigned int outH, unsigned int outW
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = N * C * outH * outW;
    if (idx >= total) return;
    unsigned int ow = idx % outW;
    unsigned int oh = (idx / outW) % outH;
    unsigned int c = (idx / (outW * outH)) % C;
    unsigned int n = idx / (outW * outH * C);
    float best = -1e38f;
    for (unsigned int kh = 0; kh < kH; kh++) {
        for (unsigned int kw = 0; kw < kW; kw++) {
            int ih = (int)(oh * strideH + kh) - (int)padH;
            int iw = (int)(ow * strideW + kw) - (int)padW;
            if (ih >= 0 && ih < (int)H && iw >= 0 && iw < (int)W) {
                float v = input[n*C*H*W + c*H*W + ih*W + iw];
                if (v > best) best = v;
            }
        }
    }
    output[idx] = best;
}
"""

NN, CC, H, W = 1, 4, 8, 8
KH, KW, SH, SW, PH, PW = 2, 2, 2, 2, 0, 0
OUT_H = (H + 2*PH - KH) // SH + 1
OUT_W = (W + 2*PW - KW) // SW + 1

def init_once():
    x = torch.randn(NN, CC, H, W, device="cuda")
    total = NN * CC * OUT_H * OUT_W
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x],
        "expected": [torch.ops.aten.max_pool2d_with_indices.default(x, [KH,KW], [SH,SW])[0].flatten()],
        "outputs": ["float32;n=%d" % total],
        "grid": ((total + 255) // 256,),
    }

def run(inputs, kernel):
    total = NN * CC * OUT_H * OUT_W
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(NN), np.uint32(CC), np.uint32(H), np.uint32(W),
        np.uint32(KH), np.uint32(KW), np.uint32(SH), np.uint32(SW),
        np.uint32(PH), np.uint32(PW), np.uint32(OUT_H), np.uint32(OUT_W),
    ])]
