"""Reference CUDA kernel for aten.sgn.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_sgn.py --once
"""
import torch

KERNEL_SRC = r"""
extern "C" __global__ void aten_sgn(const float *in0, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in0[i];
        out0[i] = ((x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f));
    }
}
"""

def init_once():
    x = torch.randn(1024, device='cuda')
    return {
        "kernel_source": KERNEL_SRC,
        "inputs": [x],
        "expected": [torch.ops.aten.sgn.default(x)],
    }

def run(inputs, kernel):
    return [kernel(*inputs)]
