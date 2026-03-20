"""Reference CUDA kernel for aten.maximum.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_maximum.py --once
"""
import torch

KERNEL_SRC = r"""
extern "C" __global__ void aten_maximum(const float *in0, const float *in1, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float a = in0[i];
        float b = in1[i];
        out0[i] = fmaxf(a, b);
    }
}
"""

def init_once():
    a = torch.randn(1024, device='cuda')
    b = torch.randn(1024, device='cuda')
    return {
        "kernel_source": KERNEL_SRC,
        "inputs": [a, b],
        "expected": [torch.ops.aten.maximum.default(a, b)],
    }

def run(inputs, kernel):
    return [kernel(*inputs)]
