"""Reference CUDA kernel for aten.round.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_round.py --once
"""
import torch

KERNEL_SRC = r"""
extern "C" __global__ void aten_round(const float *in0, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in0[i];
        out0[i] = nearbyintf(x);
    }
}
"""

def init_once():
    x = torch.randn(1024, device='cuda') * 10
    return {
        "kernel_source": KERNEL_SRC,
        "inputs": [x],
        "expected": [torch.ops.aten.round.default(x)],
    }

def run(inputs, kernel):
    return [kernel(*inputs)]
