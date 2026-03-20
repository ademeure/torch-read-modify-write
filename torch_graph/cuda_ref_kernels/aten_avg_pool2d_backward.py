"""Reference CUDA kernel for aten.avg_pool2d_backward."""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_avg_pool2d_bwd(
    const float *grad_output, float *grad_input,
    unsigned int N, unsigned int C, unsigned int H, unsigned int W,
    unsigned int kH, unsigned int kW, unsigned int strideH, unsigned int strideW,
    unsigned int padH, unsigned int padW, unsigned int outH, unsigned int outW
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = N * C * H * W;
    if (idx >= total) return;
    unsigned int iw = idx % W, ih = (idx / W) % H;
    unsigned int c = (idx / (W * H)) % C;
    unsigned int n = idx / (W * H * C);
    float sum = 0.0f;
    for (unsigned int oh = 0; oh < outH; oh++) {
        for (unsigned int ow = 0; ow < outW; ow++) {
            int h_start = (int)(oh * strideH) - (int)padH;
            int w_start = (int)(ow * strideW) - (int)padW;
            int h_end = h_start + (int)kH, w_end = w_start + (int)kW;
            if ((int)ih >= h_start && (int)ih < h_end && (int)iw >= w_start && (int)iw < w_end) {
                int count = 0;
                for (int hh = h_start; hh < h_end; hh++)
                    for (int ww = w_start; ww < w_end; ww++)
                        if (hh >= 0 && hh < (int)H && ww >= 0 && ww < (int)W) count++;
                sum += grad_output[n*C*outH*outW + c*outH*outW + oh*outW + ow] / (float)count;
            }
        }
    }
    grad_input[idx] = sum;
}
"""

NN, CC, H, W, KH, KW, SH, SW, PH, PW = 1, 4, 8, 8, 2, 2, 2, 2, 0, 0
OH, OW = (H + 2*PH - KH) // SH + 1, (W + 2*PW - KW) // SW + 1

def init_once():
    grad = torch.randn(NN, CC, OH, OW, device="cuda")
    total = NN * CC * H * W
    result = torch.ops.aten.avg_pool2d_backward.default(grad, torch.randn(NN, CC, H, W, device="cuda"), [KH,KW], [SH,SW], [PH,PW], False, True, None)
    return {"kernel_source": KERNEL_SRC, "inputs": [grad.contiguous()],
            "expected": [result.flatten()],
            "outputs": ["float32;n=%d" % total], "grid": ((total + 255) // 256,), "atol": 1e-4}

def run(inputs, kernel):
    total = NN * CC * H * W
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(NN), np.uint32(CC), np.uint32(H), np.uint32(W),
        np.uint32(KH), np.uint32(KW), np.uint32(SH), np.uint32(SW),
        np.uint32(PH), np.uint32(PW), np.uint32(OH), np.uint32(OW),
    ])]
