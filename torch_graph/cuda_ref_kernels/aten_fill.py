"""Reference CUDA kernel for aten.fill — fill tensor with scalar value.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_fill.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_fill_kernel(const float *in0, float *out0, float value, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out0[i] = value;
}
"""

def init_once():
    x = torch.randn(1024, device='cuda')
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x],
        "expected": [torch.ops.aten.fill.Scalar(x, 3.14)],
    }

def run(inputs, kernel):
    n = inputs[0].numel()
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.float32(3.14), np.uint32(n),
    ])]
