"""Reference CUDA kernel for aten.gelu_backward (backward gradient op).
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_gelu_backward.py --once
"""
import torch

KERNEL_SRC = r"""
extern "C" __global__ void aten_gelu_backward(const float *grad, const float *saved, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = grad[i];
        float s = saved[i];
        out0[i] = (g * (0.5f * (1.0f + erff(s * 0.7071067811865476f)) + s * 0.3989422804014327f * expf(-0.5f * s * s)));
    }
}
"""

def init_once():
    grad = torch.randn(1024, device='cuda')
    saved = torch.randn(1024, device='cuda')
    return {
        "kernel_source": KERNEL_SRC,
        "inputs": [grad, saved],
        "expected": [torch.ops.aten.gelu_backward.default(grad, saved)],
        "atol": 0.0001,
    }

def run(inputs, kernel):
    return [kernel(*inputs)]
