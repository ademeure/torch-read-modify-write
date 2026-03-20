"""Reference CUDA kernel for aten.lt.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_lt.py --once
"""
import torch

KERNEL_SRC = r"""
extern "C" __global__ void aten_lt(const float *in0, const float *in1, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float a = in0[i];
        float b = in1[i];
        out0[i] = (a < b ? 1.0f : 0.0f);
    }
}
"""

def init_once():
    a = torch.randn(1024, device='cuda')
    b = torch.randn(1024, device='cuda')
    return {
        "kernel_source": KERNEL_SRC,
        "inputs": [a, b],
        "expected": [torch.ops.aten.lt.Tensor(a, b).float()],
    }

def run(inputs, kernel):
    return [kernel(*inputs)]
