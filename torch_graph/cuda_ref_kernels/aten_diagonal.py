"""Reference CUDA kernel for aten.diagonal — extract diagonal."""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_diagonal(
    const float *input, float *output, unsigned int n, unsigned int cols
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = input[i * cols + i];
}
"""

def init_once():
    x = torch.randn(16, 16, device="cuda")
    n = min(x.size(0), x.size(1))
    return {"kernel_source": KERNEL_SRC, "inputs": [x.contiguous()],
            "expected": [torch.ops.aten.diagonal.default(x).contiguous()],
            "outputs": ["float32;n=%d" % n], "grid": ((n + 255) // 256,)}

def run(inputs, kernel):
    x = inputs[0]
    n = min(x.size(0), x.size(1))
    return [kernel(x, params=[kernel.in_ptr(0), kernel.out_ptr(0),
                               np.uint32(n), np.uint32(x.size(1))])]
