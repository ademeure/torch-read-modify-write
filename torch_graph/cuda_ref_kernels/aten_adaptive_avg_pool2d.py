"""Reference CUDA kernel for aten.adaptive_avg_pool2d.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_adaptive_avg_pool2d.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_adaptive_avg_pool2d_kernel(
    const float *input, float *output,
    unsigned int N, unsigned int C, unsigned int H, unsigned int W,
    unsigned int outH, unsigned int outW
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = N * C * outH * outW;
    if (idx >= total) return;
    unsigned int ow = idx % outW;
    unsigned int oh = (idx / outW) % outH;
    unsigned int c = (idx / (outW * outH)) % C;
    unsigned int n = idx / (outW * outH * C);
    unsigned int h_start = oh * H / outH;
    unsigned int h_end = (oh + 1) * H / outH;
    unsigned int w_start = ow * W / outW;
    unsigned int w_end = (ow + 1) * W / outW;
    float sum = 0.0f;
    int count = 0;
    for (unsigned int h = h_start; h < h_end; h++) {
        for (unsigned int w = w_start; w < w_end; w++) {
            sum += input[n*C*H*W + c*H*W + h*W + w];
            count++;
        }
    }
    output[idx] = sum / (float)count;
}
"""

NN, CC, H, W = 1, 4, 8, 8
OUT_H, OUT_W = 1, 1

def init_once():
    x = torch.randn(NN, CC, H, W, device="cuda")
    total = NN * CC * OUT_H * OUT_W
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x],
        "expected": [torch.ops.aten.adaptive_avg_pool2d.default(x, [OUT_H, OUT_W]).flatten()],
        "outputs": ["float32;n=%d" % total],
        "grid": ((total + 255) // 256,), "atol": 1e-4,
    }

def run(inputs, kernel):
    total = NN * CC * OUT_H * OUT_W
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(NN), np.uint32(CC), np.uint32(H), np.uint32(W),
        np.uint32(OUT_H), np.uint32(OUT_W),
    ])]
