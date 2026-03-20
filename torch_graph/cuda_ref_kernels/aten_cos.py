"""Reference CUDA kernel for aten.cos.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_cos.py --once
"""
import torch

KERNEL_SRC = r"""
extern "C" __global__ void aten_cos(const float *in0, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in0[i];
        out0[i] = cosf(x);
    }
}
"""

def init_once():
    x = torch.randn(1024, device='cuda')
    return {
        "kernel_source": KERNEL_SRC,
        "inputs": [x],
        "expected": [torch.ops.aten.cos.default(x)],
        "atol": 1e-05,
    }

def run(inputs, kernel):
    return [kernel(*inputs)]
