"""Reference CUDA kernel for aten.trunc.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_trunc.py --once
"""
import torch

KERNEL_SRC = r"""
extern "C" __global__ void aten_trunc(const float *in0, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in0[i];
        out0[i] = truncf(x);
    }
}
"""

def init_once():
    x = torch.randn(1024, device='cuda') * 10
    return {
        "kernel_source": KERNEL_SRC,
        "inputs": [x],
        "expected": [torch.ops.aten.trunc.default(x)],
    }

def run(inputs, kernel):
    return [kernel(*inputs)]
