"""Reference CUDA kernel for aten.upsample_bilinear2d."""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_upsample_bilinear2d(
    const float *input, float *output,
    unsigned int N, unsigned int C, unsigned int iH, unsigned int iW,
    unsigned int oH, unsigned int oW
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = N * C * oH * oW;
    if (idx >= total) return;
    unsigned int ow = idx % oW;
    unsigned int oh = (idx / oW) % oH;
    unsigned int c = (idx / (oW * oH)) % C;
    unsigned int n = idx / (oW * oH * C);
    // Compute source coords (align_corners=False)
    float h_scale = (float)iH / (float)oH;
    float w_scale = (float)iW / (float)oW;
    float h = ((float)oh + 0.5f) * h_scale - 0.5f;
    float w = ((float)ow + 0.5f) * w_scale - 0.5f;
    int h0 = (int)floorf(h), w0 = (int)floorf(w);
    float hf = h - h0, wf = w - w0;
    int h1 = h0 + 1, w1 = w0 + 1;
    if (h0 < 0) h0 = 0; if (h0 >= (int)iH) h0 = (int)iH - 1;
    if (h1 < 0) h1 = 0; if (h1 >= (int)iH) h1 = (int)iH - 1;
    if (w0 < 0) w0 = 0; if (w0 >= (int)iW) w0 = (int)iW - 1;
    if (w1 < 0) w1 = 0; if (w1 >= (int)iW) w1 = (int)iW - 1;
    unsigned int base = n*C*iH*iW + c*iH*iW;
    output[idx] = (1-hf)*(1-wf)*input[base + h0*iW + w0]
                + (1-hf)*wf*input[base + h0*iW + w1]
                + hf*(1-wf)*input[base + h1*iW + w0]
                + hf*wf*input[base + h1*iW + w1];
}
"""

NN, CC, IH, IW, OH, OW = 1, 4, 4, 4, 8, 8

def init_once():
    x = torch.randn(NN, CC, IH, IW, device="cuda")
    total = NN * CC * OH * OW
    return {"kernel_source": KERNEL_SRC, "inputs": [x.contiguous()],
            "expected": [torch.ops.aten.upsample_bilinear2d.vec(x, [OH, OW], False, None).flatten()],
            "outputs": ["float32;n=%d" % total], "grid": ((total + 255) // 256,), "atol": 1e-4}

def run(inputs, kernel):
    total = NN * CC * OH * OW
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(NN), np.uint32(CC), np.uint32(IH), np.uint32(IW),
        np.uint32(OH), np.uint32(OW),
    ])]
