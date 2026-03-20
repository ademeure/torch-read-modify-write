"""Reference CUDA kernel for aten._adaptive_avg_pool3d."""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_adaptive_avg_pool3d(
    const float *input, float *output,
    unsigned int N, unsigned int C,
    unsigned int D, unsigned int H, unsigned int W,
    unsigned int oD, unsigned int oH, unsigned int oW
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = N * C * oD * oH * oW;
    if (idx >= total) return;
    unsigned int ow = idx % oW, oh = (idx / oW) % oH, od = (idx / (oW * oH)) % oD;
    unsigned int c = (idx / (oW * oH * oD)) % C;
    unsigned int n = idx / (oW * oH * oD * C);
    unsigned int d0 = od*D/oD, d1 = (od+1)*D/oD;
    unsigned int h0 = oh*H/oH, h1 = (oh+1)*H/oH;
    unsigned int w0 = ow*W/oW, w1 = (ow+1)*W/oW;
    float sum = 0.0f; int count = 0;
    for (unsigned int d = d0; d < d1; d++)
        for (unsigned int h = h0; h < h1; h++)
            for (unsigned int w = w0; w < w1; w++) {
                sum += input[n*C*D*H*W + c*D*H*W + d*H*W + h*W + w]; count++;
            }
    output[idx] = sum / (float)count;
}
"""

NN, CC, DD, HH, WW, OD, OH, OW = 1, 2, 4, 4, 4, 2, 2, 2

def init_once():
    x = torch.randn(NN, CC, DD, HH, WW, device="cuda")
    total = NN * CC * OD * OH * OW
    return {"kernel_source": KERNEL_SRC, "inputs": [x.contiguous()],
            "expected": [torch.ops.aten._adaptive_avg_pool3d.default(x, [OD, OH, OW]).flatten()],
            "outputs": ["float32;n=%d" % total], "grid": ((total + 255) // 256,), "atol": 1e-5}

def run(inputs, kernel):
    total = NN * CC * OD * OH * OW
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(NN), np.uint32(CC), np.uint32(DD), np.uint32(HH), np.uint32(WW),
        np.uint32(OD), np.uint32(OH), np.uint32(OW),
    ])]
