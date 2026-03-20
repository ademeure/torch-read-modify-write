"""Reference CUDA kernel for aten.remainder.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_remainder.py --once
"""
import torch

KERNEL_SRC = r"""
extern "C" __global__ void aten_remainder(const float *in0, const float *in1, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float a = in0[i];
        float b = in1[i];
        out0[i] = (a - b * floorf(a / b));
    }
}
"""

def init_once():
    a = torch.randn(1024, device='cuda') * 10
    b = torch.randn(1024, device='cuda').abs() + 0.5
    return {
        "kernel_source": KERNEL_SRC,
        "inputs": [a, b],
        "expected": [torch.ops.aten.remainder.Tensor(a, b)],
        "atol": 0.0001,
    }

def run(inputs, kernel):
    return [kernel(*inputs)]
