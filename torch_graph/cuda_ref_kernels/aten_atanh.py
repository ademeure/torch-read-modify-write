"""Reference CUDA kernel for aten.atanh.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_atanh.py --once
"""
import torch

KERNEL_SRC = r"""
extern "C" __global__ void aten_atanh(const float *in0, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in0[i];
        out0[i] = atanhf(x);
    }
}
"""

def init_once():
    x = torch.rand(1024, device='cuda') * 1.98 - 0.99
    return {
        "kernel_source": KERNEL_SRC,
        "inputs": [x],
        "expected": [torch.ops.aten.atanh.default(x)],
        "atol": 1e-05,
    }

def run(inputs, kernel):
    return [kernel(*inputs)]
