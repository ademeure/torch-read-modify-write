"""Reference CUDA kernel for aten.tril — lower triangle of a matrix.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_tril.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_tril_kernel(
    const float *input, float *output, unsigned int rows, unsigned int cols, int diagonal
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    unsigned int r = idx / cols, c = idx % cols;
    output[idx] = ((int)c <= (int)r + diagonal) ? input[idx] : 0.0f;
}
"""

N = 16

def init_once():
    x = torch.randn(N, N, device="cuda")
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x],
        "expected": [torch.ops.aten.tril.default(x)],
    }

def run(inputs, kernel):
    n = inputs[0].size(0)
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(n), np.uint32(n), np.int32(0),
    ])]
