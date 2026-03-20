"""Reference CUDA kernel for aten.gelu_backward (backward gradient op)."""
import torch

KERNEL_SRC = r"""
extern "C" __global__ void aten_gelu_backward(const float *grad, const float *saved, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float g = grad[i], s = saved[i]; out0[i] = (g * (0.5f * (1.0f + erff(s * 0.7071067811865476f)) + s * 0.3989422804014327f * expf(-0.5f * s * s))); }
}
"""

ATOL = 0.0001

def make_inputs(n=1024, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    grad = torch.randn(1024, device="cuda", generator=g)
    saved = torch.randn(1024, device="cuda", generator=g)
    return [grad, saved]

def expected(inputs):
    grad, saved = inputs
    return [torch.ops.aten.gelu_backward.default(grad, saved)]

def init_once():
    inputs = make_inputs()
    return {"kernel_source": KERNEL_SRC, "inputs": inputs, "expected": expected(inputs), "atol": ATOL}

def run(inputs, kernel):
    return [kernel(*inputs)]
