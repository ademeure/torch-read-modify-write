"""Reference CUDA kernel for aten.remainder."""
import torch

KERNEL_SRC = r"""
extern "C" __global__ void aten_remainder(const float *in0, const float *in1, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float a = in0[i], b = in1[i]; out0[i] = (a - b * floorf(a / b)); }
}
"""

ATOL = 0.0001

def make_inputs(n=1024, seed=1):
    if seed == 0:
        special = torch.tensor([0.0, -0.0, 1.0, -1.0, 0.5, 2.0, 4.0, 100.0, -100.0, 1e-7, 1e7, float("nan"), float("inf"), float("-inf")], device="cuda")
        s = special.repeat((n + len(special) - 1) // len(special))[:n]
        return [s, s.flip(0)]
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn(1024, device="cuda", generator=g) * 10
    b = torch.randn(1024, device="cuda", generator=g).abs() + 0.5
    return [a, b]

def expected(inputs):
    a, b = inputs
    return [torch.ops.aten.remainder.Tensor(a, b)]

def init_once():
    inputs = make_inputs()
    return {"kernel_source": KERNEL_SRC, "inputs": inputs, "expected": expected(inputs), "atol": ATOL}

def run(inputs, kernel):
    return [kernel(*inputs)]
