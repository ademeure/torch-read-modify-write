"""Reference CUDA kernel for aten.avg_pool1d."""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_avg_pool1d(
    const float *input, float *output,
    unsigned int N, unsigned int C, unsigned int L,
    unsigned int kL, unsigned int stride, unsigned int pad, unsigned int outL
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = N * C * outL;
    if (idx >= total) return;
    unsigned int ol = idx % outL;
    unsigned int c = (idx / outL) % C;
    unsigned int n = idx / (outL * C);
    float sum = 0.0f; int count = 0;
    for (unsigned int k = 0; k < kL; k++) {
        int il = (int)(ol * stride + k) - (int)pad;
        if (il >= 0 && il < (int)L) { sum += input[n*C*L + c*L + il]; count++; }
    }
    output[idx] = sum / (float)kL;
}
"""

NN, CC, LL, KL, ST, PAD = 2, 4, 16, 3, 1, 1
OL = (LL + 2*PAD - KL) // ST + 1

def init_once():
    x = torch.randn(NN, CC, LL, device="cuda")
    total = NN * CC * OL
    return {"kernel_source": KERNEL_SRC, "inputs": [x.contiguous()],
            "expected": [torch.ops.aten.avg_pool1d.default(x, [KL], [ST], [PAD]).flatten()],
            "outputs": ["float32;n=%d" % total], "grid": ((total + 255) // 256,), "atol": 1e-5}

def run(inputs, kernel):
    total = NN * CC * OL
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(NN), np.uint32(CC), np.uint32(LL),
        np.uint32(KL), np.uint32(ST), np.uint32(PAD), np.uint32(OL),
    ])]
