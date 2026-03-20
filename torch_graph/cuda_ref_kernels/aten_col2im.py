"""Reference CUDA kernel for aten.col2im."""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_col2im(
    const float *col, float *im,
    unsigned int N, unsigned int C, unsigned int H, unsigned int W,
    unsigned int kH, unsigned int kW, unsigned int pH, unsigned int pW,
    unsigned int sH, unsigned int sW, unsigned int dH, unsigned int dW,
    unsigned int outH, unsigned int outW
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = N * C * H * W;
    if (idx >= total) return;
    im[idx] = 0.0f;  // zero init
    unsigned int iw = idx % W, ih = (idx / W) % H;
    unsigned int c = (idx / (W * H)) % C;
    unsigned int n = idx / (W * H * C);
    unsigned int col_C = C * kH * kW;
    for (unsigned int kh = 0; kh < kH; kh++) {
        for (unsigned int kw = 0; kw < kW; kw++) {
            int oh = ((int)ih + (int)pH - (int)(kh * dH));
            int ow_val = ((int)iw + (int)pW - (int)(kw * dW));
            if (oh % (int)sH == 0 && ow_val % (int)sW == 0) {
                oh /= (int)sH; ow_val /= (int)sW;
                if (oh >= 0 && oh < (int)outH && ow_val >= 0 && ow_val < (int)outW) {
                    unsigned int col_idx = c*kH*kW + kh*kW + kw;
                    im[idx] += col[n*col_C*outH*outW + col_idx*outH*outW + oh*outW + ow_val];
                }
            }
        }
    }
}
"""

def init_once():
    # Simple case: 1x1 kernel = identity
    col = torch.randn(1, 4, 16, device="cuda")  # N=1, C*kH*kW=4, L=16
    H, W = 4, 4
    total = 1 * 4 * H * W
    result = torch.ops.aten.col2im.default(col, [H, W], [1, 1], [1, 1], [0, 0], [1, 1])
    return {"kernel_source": KERNEL_SRC, "inputs": [col.contiguous()],
            "expected": [result.flatten()],
            "outputs": ["float32;n=%d" % total], "grid": ((total + 255) // 256,), "atol": 1e-4}

def run(inputs, kernel):
    total = 1 * 4 * 4 * 4
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(1), np.uint32(4), np.uint32(4), np.uint32(4),
        np.uint32(1), np.uint32(1), np.uint32(0), np.uint32(0),
        np.uint32(1), np.uint32(1), np.uint32(1), np.uint32(1),
        np.uint32(4), np.uint32(4),
    ])]
