"""Reference CUDA kernel for aten.gelu.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_gelu.py --once
"""
import torch

KERNEL_SRC = r"""
extern "C" __global__ void aten_gelu(const float *in0, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in0[i];
        out0[i] = (x * 0.5f * (1.0f + erff(x * 0.7071067811865476f)));
    }
}
"""

def init_once():
    x = torch.randn(1024, device='cuda')
    return {
        "kernel_source": KERNEL_SRC,
        "inputs": [x],
        "expected": [torch.ops.aten.gelu.default(x)],
        "atol": 1e-05,
    }

def run(inputs, kernel):
    return [kernel(*inputs)]
