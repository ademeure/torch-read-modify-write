"""Reference CUDA kernel for aten.sqrt.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_sqrt.py --once
"""
import torch

KERNEL_SRC = r"""
extern "C" __global__ void aten_sqrt(const float *in0, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in0[i];
        out0[i] = sqrtf(x);
    }
}
"""

def init_once():
    x = torch.rand(1024, device='cuda') + 0.01
    return {
        "kernel_source": KERNEL_SRC,
        "inputs": [x],
        "expected": [torch.ops.aten.sqrt.default(x)],
        "atol": 1e-06,
    }

def run(inputs, kernel):
    return [kernel(*inputs)]
