"""Reference CUDA kernel for aten.ge."""
import torch

KERNEL_SRC = r"""
extern "C" __global__ void aten_ge(const float *in0, const float *in1, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float a = in0[i], b = in1[i]; out0[i] = (a >= b ? 1.0f : 0.0f); }
}
"""

def make_inputs(n=1024, seed=1):
    if seed == 0:
        special = torch.tensor([0.0, -0.0, 1.0, -1.0, 0.5, 2.0, 100.0, -100.0, float("nan"), float("inf"), float("-inf")], device="cuda")
        s = special.repeat((n + len(special) - 1) // len(special))[:n]
        return [s, s.flip(0)]
    g = torch.Generator(device="cuda").manual_seed(seed)
    return [torch.randn(n, device="cuda", generator=g), torch.randn(n, device="cuda", generator=g)]

def expected(inputs):
    a, b = inputs
    return [torch.ops.aten.ge.Tensor(a, b).float()]

def init_once():
    inputs = make_inputs()
    return {"kernel_source": KERNEL_SRC, "inputs": inputs, "expected": expected(inputs)}

def run(inputs, kernel):
    return [kernel(*inputs)]
