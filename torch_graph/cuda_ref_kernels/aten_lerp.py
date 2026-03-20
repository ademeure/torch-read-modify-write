"""Reference CUDA kernel for aten.lerp — linear interpolation.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_lerp.py --once
"""
import torch

KERNEL_SRC = r"""
extern "C" __global__ void aten_lerp(
    const float *a, const float *b, const float *w, float *out, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + w[i] * (b[i] - a[i]);
}
"""

def init_once():
    a = torch.randn(1024, device='cuda')
    b = torch.randn(1024, device='cuda')
    w = torch.rand(1024, device='cuda')
    return {
        "kernel_source": KERNEL_SRC, "inputs": [a, b, w],
        "expected": [torch.ops.aten.lerp.Tensor(a, b, w)], "atol": 1e-5,
    }

def run(inputs, kernel):
    return [kernel(*inputs)]
