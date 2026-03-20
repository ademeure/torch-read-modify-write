"""Reference CUDA kernel for aten.remainder.Scalar."""
import torch

KERNEL_SRC = r"""
extern "C" __global__ void aten_remainder_scalar(const float *in0, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float a = in0[i], b = 2.0f;
        float r = fmodf(a, b);
        if (r != 0.0f && ((r < 0.0f) != (b < 0.0f))) r += b;
        out0[i] = r;
    }
}
"""

def init_once():
    x = torch.randn(1024, device="cuda") * 10
    return {"kernel_source": KERNEL_SRC, "inputs": [x],
            "expected": [torch.ops.aten.remainder.Scalar(x, 2.0)], "atol": 1e-4}

def run(inputs, kernel):
    return [kernel(*inputs)]
