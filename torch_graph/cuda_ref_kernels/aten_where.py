"""Reference CUDA kernel for aten.where — elementwise conditional select."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_where(
    const float *cond, const float *x, const float *y, float *out, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = (cond[i] != 0.0f) ? x[i] : y[i];
    }
}

torch::Tensor aten_where_fwd(torch::Tensor cond, torch::Tensor x, torch::Tensor y) {
    auto out = torch::empty_like(x);
    int n = x.numel();
    aten_where<<<(n+255)/256, 256>>>(
        cond.data_ptr<float>(), x.data_ptr<float>(),
        y.data_ptr<float>(), out.data_ptr<float>(), n);
    return out;
}
"""

def test():
    ext = compile_cuda("aten_where", KERNEL_SRC, ["aten_where_fwd"])
    cond = (torch.randn(1024, device='cuda') > 0).float()
    x = torch.randn(1024, device='cuda')
    y = torch.randn(1024, device='cuda')
    result = ext.aten_where_fwd(cond, x, y)
    expected = aten.where.self(cond.bool(), x, y)
    check("aten.where", result, expected)
    print("PASS aten.where")

if __name__ == "__main__":
    test()
