"""Reference CUDA kernel for aten.acosh.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_acosh.py --once
"""
import torch

KERNEL_SRC = r"""
extern "C" __global__ void aten_acosh(const float *in0, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in0[i];
        out0[i] = acoshf(x);
    }
}
"""

def init_once():
    x = torch.rand(1024, device='cuda') + 1.01
    return {
        "kernel_source": KERNEL_SRC,
        "inputs": [x],
        "expected": [torch.ops.aten.acosh.default(x)],
        "atol": 1e-05,
    }

def run(inputs, kernel):
    return [kernel(*inputs)]
