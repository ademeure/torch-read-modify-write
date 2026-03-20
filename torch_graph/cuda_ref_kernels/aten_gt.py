"""Reference CUDA kernel for aten.gt."""
import torch

KERNEL_SRC = r"""
extern "C" __global__ void aten_gt(const float *in0, const float *in1, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float a = in0[i], b = in1[i]; out0[i] = (a > b ? 1.0f : 0.0f); }
}
"""

def make_inputs(n=1024, seed=1):
    if seed == 0:
        special = torch.tensor([0.0, -0.0, 1.0, -1.0, 0.5, 2.0, 100.0, -100.0, 1e-45, -1e-45, 1.18e-38, float("nan"), float("inf"), float("-inf")], device="cuda")
        m = len(special)
        a = special.repeat_interleave(m).repeat((n + m*m - 1) // (m*m))[:n]
        b = special.repeat(m).repeat((n + m*m - 1) // (m*m))[:n]
        return [a, b]
    g = torch.Generator(device="cuda").manual_seed(seed)
    return [torch.randn(n, device="cuda", generator=g), torch.randn(n, device="cuda", generator=g)]

def expected(inputs):
    a, b = inputs
    return [torch.ops.aten.gt.Tensor(a, b).float()]

def init_once():
    inputs = make_inputs()
    return {"kernel_source": KERNEL_SRC, "inputs": inputs, "expected": expected(inputs)}

def run(inputs, kernel):
    return [kernel(*inputs)]
