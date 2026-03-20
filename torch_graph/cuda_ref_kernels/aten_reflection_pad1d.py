"""Reference CUDA kernel for aten.reflection_pad1d."""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_reflection_pad1d(
    const float *input, float *output,
    unsigned int N, unsigned int C, unsigned int L, unsigned int outL, unsigned int padL
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = N * C * outL;
    if (idx >= total) return;
    unsigned int ol = idx % outL;
    unsigned int c = (idx / outL) % C;
    unsigned int n = idx / (outL * C);
    int il = (int)ol - (int)padL;
    if (il < 0) il = -il;
    if (il >= (int)L) il = 2*(int)L - il - 2;
    output[idx] = input[n*C*L + c*L + il];
}
"""

NN, CC, LL, PADL, PADR = 2, 4, 16, 3, 3
OL = LL + PADL + PADR

def init_once():
    x = torch.randn(NN, CC, LL, device="cuda")
    total = NN * CC * OL
    return {"kernel_source": KERNEL_SRC, "inputs": [x.contiguous()],
            "expected": [torch.ops.aten.reflection_pad1d.default(x, [PADL, PADR]).flatten()],
            "outputs": ["float32;n=%d" % total], "grid": ((total + 255) // 256,)}

def run(inputs, kernel):
    total = NN * CC * OL
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(NN), np.uint32(CC), np.uint32(LL), np.uint32(OL), np.uint32(PADL),
    ])]
