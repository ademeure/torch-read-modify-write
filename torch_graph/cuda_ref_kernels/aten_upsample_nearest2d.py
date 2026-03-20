"""Reference CUDA kernel for aten.upsample_nearest2d."""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_upsample_nearest2d(
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
    unsigned int ih = oh * iH / oH;
    unsigned int iw = ow * iW / oW;
    output[idx] = input[n*C*iH*iW + c*iH*iW + ih*iW + iw];
}
"""

NN, CC, IH, IW, OH, OW = 1, 4, 4, 4, 8, 8

def init_once():
    x = torch.randn(NN, CC, IH, IW, device="cuda")
    total = NN * CC * OH * OW
    return {"kernel_source": KERNEL_SRC, "inputs": [x.contiguous()],
            "expected": [torch.ops.aten.upsample_nearest2d.vec(x, [OH, OW], None).flatten()],
            "outputs": ["float32;n=%d" % total], "grid": ((total + 255) // 256,)}

def run(inputs, kernel):
    total = NN * CC * OH * OW
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(NN), np.uint32(CC), np.uint32(IH), np.uint32(IW),
        np.uint32(OH), np.uint32(OW),
    ])]
