"""Reference CUDA kernel for aten.tan.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_tan.py --once
"""
import torch

KERNEL_SRC = r"""
extern "C" __global__ void aten_tan(const float *in0, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in0[i];
        out0[i] = tanf(x);
    }
}
"""

def init_once():
    x = torch.randn(1024, device='cuda') * 0.5
    return {
        "kernel_source": KERNEL_SRC,
        "inputs": [x],
        "expected": [torch.ops.aten.tan.default(x)],
        "atol": 0.0001,
    }

def run(inputs, kernel):
    return [kernel(*inputs)]
