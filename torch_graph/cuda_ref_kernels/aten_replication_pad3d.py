"""Reference CUDA kernel for aten.replication_pad3d."""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_replication_pad3d(
    const float *input, float *output,
    unsigned int N, unsigned int C,
    unsigned int D, unsigned int H, unsigned int W,
    unsigned int oD, unsigned int oH, unsigned int oW,
    unsigned int pD, unsigned int pH, unsigned int pW
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = N * C * oD * oH * oW;
    if (idx >= total) return;
    unsigned int ow = idx % oW, oh = (idx / oW) % oH, od = (idx / (oW*oH)) % oD;
    unsigned int c = (idx / (oW*oH*oD)) % C;
    unsigned int n = idx / (oW*oH*oD*C);
    int id = (int)od-(int)pD, ih = (int)oh-(int)pH, iw = (int)ow-(int)pW;
    if (id < 0) id = 0; if (id >= (int)D) id = D-1;
    if (ih < 0) ih = 0; if (ih >= (int)H) ih = H-1;
    if (iw < 0) iw = 0; if (iw >= (int)W) iw = W-1;
    output[idx] = input[n*C*D*H*W + c*D*H*W + id*H*W + ih*W + iw];
}
"""

NN, CC, DD, HH, WW, PAD = 1, 2, 4, 4, 4, 1
OD, OH, OW = DD+2*PAD, HH+2*PAD, WW+2*PAD

def init_once():
    x = torch.randn(NN, CC, DD, HH, WW, device="cuda")
    total = NN * CC * OD * OH * OW
    return {"kernel_source": KERNEL_SRC, "inputs": [x.contiguous()],
            "expected": [torch.ops.aten.replication_pad3d.default(x, [PAD]*6).flatten()],
            "outputs": ["float32;n=%d" % total], "grid": ((total + 255) // 256,)}

def run(inputs, kernel):
    total = NN * CC * OD * OH * OW
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(NN), np.uint32(CC), np.uint32(DD), np.uint32(HH), np.uint32(WW),
        np.uint32(OD), np.uint32(OH), np.uint32(OW),
        np.uint32(PAD), np.uint32(PAD), np.uint32(PAD),
    ])]
