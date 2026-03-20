"""Reference CUDA kernel for aten.scatter_reduce — scatter with reduction."""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_scatter_reduce_init(const float *self, float *out, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = self[i];
}
extern "C" __global__ void aten_scatter_reduce_add(
    const long *index, const float *src, float *out,
    unsigned int rows, unsigned int in_cols, unsigned int src_cols
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * src_cols) return;
    unsigned int r = idx / src_cols, c = idx % src_cols;
    long dst_c = index[idx];
    atomicAdd(&out[r * in_cols + dst_c], src[idx]);
}
"""

def init_once():
    x = torch.zeros(8, 32, device="cuda")
    idx = torch.randint(0, 32, (8, 16), device="cuda")
    src = torch.randn(8, 16, device="cuda")
    total = x.numel()
    return {"kernel_source": KERNEL_SRC, "inputs": [x, idx, src],
            "expected": [torch.ops.aten.scatter_reduce.two(x, 1, idx, src, "sum").flatten()],
            "outputs": ["float32;n=%d" % total], "grid": ((total + 255) // 256,), "atol": 1e-4}

def run(inputs, kernel):
    return [kernel(*inputs)]
