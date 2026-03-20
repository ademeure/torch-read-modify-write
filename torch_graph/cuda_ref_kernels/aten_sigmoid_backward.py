"""Reference CUDA kernel for aten.sigmoid_backward (backward gradient op).
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_sigmoid_backward.py --once
"""
import torch

KERNEL_SRC = r"""
extern "C" __global__ void aten_sigmoid_backward(const float *grad, const float *saved, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = grad[i];
        float s = saved[i];
        out0[i] = (g * s * (1.0f - s));
    }
}
"""

def init_once():
    grad = torch.randn(1024, device='cuda')
    saved = torch.sigmoid(torch.randn(1024, device='cuda'))
    return {
        "kernel_source": KERNEL_SRC,
        "inputs": [grad, saved],
        "expected": [torch.ops.aten.sigmoid_backward.default(grad, saved)],
        "atol": 1e-05,
    }

def run(inputs, kernel):
    return [kernel(*inputs)]
