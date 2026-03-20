"""Reference CUDA kernel for aten.threshold_backward (backward gradient op).
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_threshold_backward.py --once
"""
import torch

KERNEL_SRC = r"""
extern "C" __global__ void aten_threshold_backward(const float *grad, const float *saved, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = grad[i];
        float s = saved[i];
        out0[i] = (s > 0.0f ? g : 0.0f);
    }
}
"""

def init_once():
    grad = torch.randn(1024, device='cuda')
    saved = torch.randn(1024, device='cuda')
    return {
        "kernel_source": KERNEL_SRC,
        "inputs": [grad, saved],
        "expected": [torch.ops.aten.threshold_backward.default(grad, saved, 0.0)],
        "atol": 1e-05,
    }

def run(inputs, kernel):
    return [kernel(*inputs)]
