"""Reference CUDA kernel for aten.where — elementwise conditional select.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_where.py --once
"""
import torch

KERNEL_SRC = r"""
extern "C" __global__ void aten_where(
    const float *cond, const float *x, const float *y, float *out, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (cond[i] != 0.0f) ? x[i] : y[i];
}
"""

def init_once():
    cond = (torch.randn(1024, device='cuda') > 0).float()
    x = torch.randn(1024, device='cuda')
    y = torch.randn(1024, device='cuda')
    return {
        "kernel_source": KERNEL_SRC, "inputs": [cond, x, y],
        "expected": [torch.ops.aten.where.self(cond.bool(), x, y)],
    }

def run(inputs, kernel):
    return [kernel(*inputs)]
