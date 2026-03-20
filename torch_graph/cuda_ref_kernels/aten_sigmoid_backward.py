"""Reference CUDA kernel for aten.sigmoid_backward (backward gradient op)."""
import torch

KERNEL_SRC = r"""
extern "C" __global__ void aten_sigmoid_backward(const float *grad, const float *saved, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float g = grad[i], s = saved[i]; out0[i] = (g * s * (1.0f - s)); }
}
"""

ATOL = 1e-05

def make_inputs(n=1024, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    grad = torch.randn(1024, device="cuda", generator=g)
    saved = torch.sigmoid(torch.randn(1024, device="cuda", generator=g))
    return [grad, saved]

def expected(inputs):
    grad, saved = inputs
    return [torch.ops.aten.sigmoid_backward.default(grad, saved)]

def init_once():
    inputs = make_inputs()
    return {"kernel_source": KERNEL_SRC, "inputs": inputs, "expected": expected(inputs), "atol": ATOL}

def run(inputs, kernel):
    return [kernel(*inputs)]
