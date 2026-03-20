"""Reference CUDA kernel for aten.cat — concatenate tensors along a dimension."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

// Cat two 2D tensors along dim=0
extern "C" __global__ void aten_cat_dim0(
    const float *a, const float *b, float *out,
    unsigned int a_rows, unsigned int b_rows, unsigned int cols
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = (a_rows + b_rows) * cols;
    if (idx >= total) return;
    unsigned int r = idx / cols, c = idx % cols;
    if (r < a_rows)
        out[idx] = a[r * cols + c];
    else
        out[idx] = b[(r - a_rows) * cols + c];
}

torch::Tensor aten_cat_dim0_fwd(torch::Tensor a, torch::Tensor b) {
    auto ca = a.contiguous(), cb = b.contiguous();
    int ar = ca.size(0), br = cb.size(0), cols = ca.size(1);
    auto out = torch::empty({ar + br, cols}, ca.options());
    int total = (ar + br) * cols;
    aten_cat_dim0<<<(total+255)/256, 256>>>(
        ca.data_ptr<float>(), cb.data_ptr<float>(), out.data_ptr<float>(),
        ar, br, cols);
    return out;
}
"""

def test():
    ext = compile_cuda("aten_cat_dim0", KERNEL_SRC, ["aten_cat_dim0_fwd"])
    a = torch.randn(8, 32, device="cuda")
    b = torch.randn(16, 32, device="cuda")
    result = ext.aten_cat_dim0_fwd(a, b)
    expected = aten.cat.default([a, b], 0)
    check("aten.cat", result, expected)
    print("PASS aten.cat")

if __name__ == "__main__":
    test()
