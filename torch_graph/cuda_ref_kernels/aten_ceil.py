"""Reference CUDA kernel for aten.ceil.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_ceil.py --once
"""
import torch

KERNEL_SRC = r"""
extern "C" __global__ void aten_ceil(const float *in0, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in0[i];
        out0[i] = ceilf(x);
    }
}
"""

def init_once():
    x = torch.randn(1024, device='cuda') * 10
    return {
        "kernel_source": KERNEL_SRC,
        "inputs": [x],
        "expected": [torch.ops.aten.ceil.default(x)],
    }

def run(inputs, kernel):
    return [kernel(*inputs)]
