"""Reference CUDA kernel for aten._adaptive_avg_pool2d_backward."""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_adaptive_avg_pool2d_bwd(
    const float *grad_output, float *grad_input,
    unsigned int N, unsigned int C, unsigned int H, unsigned int W,
    unsigned int outH, unsigned int outW
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = N * C * H * W;
    if (idx >= total) return;
    unsigned int iw = idx % W, ih = (idx / W) % H;
    unsigned int c = (idx / (W * H)) % C;
    unsigned int n = idx / (W * H * C);
    float sum = 0.0f;
    for (unsigned int oh = 0; oh < outH; oh++) {
        unsigned int h_start = oh * H / outH, h_end = (oh + 1) * H / outH;
        if (ih < h_start || ih >= h_end) continue;
        for (unsigned int ow = 0; ow < outW; ow++) {
            unsigned int w_start = ow * W / outW, w_end = (ow + 1) * W / outW;
            if (iw < w_start || iw >= w_end) continue;
            unsigned int count = (h_end - h_start) * (w_end - w_start);
            sum += grad_output[n*C*outH*outW + c*outH*outW + oh*outW + ow] / (float)count;
        }
    }
    grad_input[idx] = sum;
}
"""

def init_once():
    x = torch.randn(1, 4, 8, 8, device="cuda")
    grad = torch.randn(1, 4, 1, 1, device="cuda")
    result = torch.ops.aten._adaptive_avg_pool2d_backward.default(grad, x)
    total = x.numel()
    return {"kernel_source": KERNEL_SRC, "inputs": [grad.contiguous()],
            "expected": [result.flatten()],
            "outputs": ["float32;n=%d" % total], "grid": ((total + 255) // 256,), "atol": 1e-4}

def run(inputs, kernel):
    total = 1 * 4 * 8 * 8
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(1), np.uint32(4), np.uint32(8), np.uint32(8),
        np.uint32(1), np.uint32(1),
    ])]
