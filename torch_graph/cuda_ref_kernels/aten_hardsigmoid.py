"""Reference CUDA kernel for aten.hardsigmoid."""
import torch

KERNEL_SRC = r"""
extern "C" __global__ void aten_hardsigmoid(const float *in0, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float x = in0[i]; out0[i] = fminf(fmaxf(x / 6.0f + 0.5f, 0.0f), 1.0f); }
}
"""

ATOL = 1e-05

def make_inputs(n=1024, seed=1):
    """seed=0 → special values (nan/inf/0/1/etc), seed>0 → seeded random."""
    if seed == 0:
        special = torch.tensor([0.0, -0.0, 1.0, -1.0, 0.5, 2.0, 4.0, 100.0, -100.0, 1e-7, 1e7, float("nan"), float("inf"), float("-inf")], device="cuda")
        return [special.repeat((n + len(special) - 1) // len(special))[:n]]
    g = torch.Generator(device="cuda").manual_seed(seed)
    return [torch.randn(n, device="cuda", generator=g) * 5]

def expected(inputs):
    x = inputs[0]
    return [torch.ops.aten.hardsigmoid.default(inputs[0])]

def init_once():
    inputs = make_inputs()
    return {"kernel_source": KERNEL_SRC, "inputs": inputs, "expected": expected(inputs), "atol": ATOL}

def run(inputs, kernel):
    return [kernel(*inputs)]
