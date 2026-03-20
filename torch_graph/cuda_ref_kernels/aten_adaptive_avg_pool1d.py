"""Reference CUDA kernel for aten.adaptive_avg_pool1d."""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_adaptive_avg_pool1d(
    const float *input, float *output,
    unsigned int N, unsigned int C, unsigned int L, unsigned int outL
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = N * C * outL;
    if (idx >= total) return;
    unsigned int ol = idx % outL;
    unsigned int c = (idx / outL) % C;
    unsigned int n = idx / (outL * C);
    unsigned int l_start = ol * L / outL, l_end = (ol + 1) * L / outL;
    float sum = 0.0f;
    for (unsigned int l = l_start; l < l_end; l++) sum += input[n*C*L + c*L + l];
    output[idx] = sum / (float)(l_end - l_start);
}
"""

NN, CC, LL, OL = 2, 4, 16, 4

def init_once():
    x = torch.randn(NN, CC, LL, device="cuda")
    total = NN * CC * OL
    return {"kernel_source": KERNEL_SRC, "inputs": [x.contiguous()],
            "expected": [torch.ops.aten.adaptive_avg_pool1d.default(x, [OL]).flatten()],
            "outputs": ["float32;n=%d" % total], "grid": ((total + 255) // 256,), "atol": 1e-5}

def run(inputs, kernel):
    total = NN * CC * OL
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(NN), np.uint32(CC), np.uint32(LL), np.uint32(OL),
    ])]
