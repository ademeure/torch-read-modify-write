"""Reference CUDA kernel for aten.max_pool3d_with_indices."""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_max_pool3d(
    const float *input, float *output,
    unsigned int N, unsigned int C,
    unsigned int D, unsigned int H, unsigned int W,
    unsigned int kD, unsigned int kH, unsigned int kW,
    unsigned int sD, unsigned int sH, unsigned int sW,
    unsigned int pD, unsigned int pH, unsigned int pW,
    unsigned int oD, unsigned int oH, unsigned int oW
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = N * C * oD * oH * oW;
    if (idx >= total) return;
    unsigned int ow = idx % oW, oh = (idx / oW) % oH, od = (idx / (oW*oH)) % oD;
    unsigned int c = (idx / (oW*oH*oD)) % C;
    unsigned int n = idx / (oW*oH*oD*C);
    float best = -1e38f;
    for (unsigned int kd = 0; kd < kD; kd++)
        for (unsigned int kh = 0; kh < kH; kh++)
            for (unsigned int kw = 0; kw < kW; kw++) {
                int id=(int)(od*sD+kd)-(int)pD, ih=(int)(oh*sH+kh)-(int)pH, iw=(int)(ow*sW+kw)-(int)pW;
                if (id>=0&&id<(int)D&&ih>=0&&ih<(int)H&&iw>=0&&iw<(int)W) {
                    float v = input[n*C*D*H*W + c*D*H*W + id*H*W + ih*W + iw];
                    if (v > best) best = v;
                }
            }
    output[idx] = best;
}
"""

NN, CC, DD, HH, WW = 1, 2, 4, 4, 4
KD, KH, KW, SD, SH, SW, PD, PH, PW = 2, 2, 2, 2, 2, 2, 0, 0, 0
OD, OH, OW = (DD+2*PD-KD)//SD+1, (HH+2*PH-KH)//SH+1, (WW+2*PW-KW)//SW+1

def init_once():
    x = torch.randn(NN, CC, DD, HH, WW, device="cuda")
    total = NN * CC * OD * OH * OW
    return {"kernel_source": KERNEL_SRC, "inputs": [x.contiguous()],
            "expected": [torch.ops.aten.max_pool3d_with_indices.default(x, [KD,KH,KW], [SD,SH,SW])[0].flatten()],
            "outputs": ["float32;n=%d" % total], "grid": ((total + 255) // 256,), "atol": 1e-5}

def run(inputs, kernel):
    total = NN * CC * OD * OH * OW
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(NN), np.uint32(CC), np.uint32(DD), np.uint32(HH), np.uint32(WW),
        np.uint32(KD), np.uint32(KH), np.uint32(KW),
        np.uint32(SD), np.uint32(SH), np.uint32(SW),
        np.uint32(PD), np.uint32(PH), np.uint32(PW),
        np.uint32(OD), np.uint32(OH), np.uint32(OW),
    ])]
