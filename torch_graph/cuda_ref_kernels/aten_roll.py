"""Reference CUDA kernel for aten.roll — circular shift.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_roll.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_roll_1d(
    const float *input, float *output, unsigned int n, int shift
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int src = ((int)i - shift % (int)n + (int)n) % (int)n;
    output[i] = input[src];
}
"""

def init_once():
    x = torch.randn(256, device="cuda")
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x],
        "expected": [torch.ops.aten.roll.default(x, [10]).contiguous()],
    }

def run(inputs, kernel):
    n = inputs[0].numel()
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(n), np.int32(10),
    ])]
