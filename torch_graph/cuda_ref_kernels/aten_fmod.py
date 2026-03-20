"""Reference CUDA kernel for aten.fmod."""
import torch

KERNEL_SRC = r"""
extern "C" __global__ void aten_fmod(const float *in0, const float *in1, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float a = in0[i], b = in1[i]; out0[i] = fmodf(a, b); }
}
"""

ATOL = 1e-05

def make_inputs(n=1024, seed=1):
    if seed == 0:
        # All special values paired with every other special value (cross-product)
        special = torch.tensor([0.0, -0.0, 1.0, -1.0, 0.5, 2.0, 4.0, 100.0, -100.0, 1e-7, 1e7, 1e-45, -1e-45, 1.18e-38, -1.18e-38, float("nan"), float("inf"), float("-inf")], device="cuda")
        m = len(special)
        a = special.repeat_interleave(m).repeat((n + m*m - 1) // (m*m))[:n]
        b = special.repeat(m).repeat((n + m*m - 1) // (m*m))[:n]
        return [a, b]
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn(1024, device="cuda", generator=g) * 10
    b = torch.randn(1024, device="cuda", generator=g).abs() + 0.5
    return [a, b]

def expected(inputs):
    a, b = inputs
    return [torch.ops.aten.fmod.Tensor(a, b)]

def init_once():
    inputs = make_inputs()
    return {"kernel_source": KERNEL_SRC, "inputs": inputs, "expected": expected(inputs), "atol": ATOL}

def run(inputs, kernel):
    return [kernel(*inputs)]
