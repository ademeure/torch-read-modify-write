"""Reference CUDA kernel for aten.rsqrt.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_rsqrt.py --once
"""
import torch

KERNEL_SRC = r"""
extern "C" __global__ void aten_rsqrt(const float *in0, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in0[i];
        out0[i] = rsqrtf(x);
    }
}
"""

def init_once():
    x = torch.rand(1024, device='cuda') + 0.01
    return {
        "kernel_source": KERNEL_SRC,
        "inputs": [x],
        "expected": [torch.ops.aten.rsqrt.default(x)],
        "atol": 0.0001,
    }

def run(inputs, kernel):
    return [kernel(*inputs)]
