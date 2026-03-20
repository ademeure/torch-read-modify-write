"""Reference CUDA kernel for aten.stack — stack tensors along new dimension."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

// Stack N 1D tensors → 2D tensor [N, L]
extern "C" __global__ void aten_stack_2(
    const float *a, const float *b, float *out, unsigned int L
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < L) {
        out[i] = a[i];
        out[L + i] = b[i];
    }
}

torch::Tensor aten_stack_2_fwd(torch::Tensor a, torch::Tensor b) {
    auto ca = a.contiguous(), cb = b.contiguous();
    int L = ca.numel();
    auto out = torch::empty({2, L}, ca.options());
    aten_stack_2<<<(L+255)/256, 256>>>(
        ca.data_ptr<float>(), cb.data_ptr<float>(), out.data_ptr<float>(), L);
    return out;
}
"""

def test():
    ext = compile_cuda("aten_stack_2", KERNEL_SRC, ["aten_stack_2_fwd"])
    a = torch.randn(64, device="cuda")
    b = torch.randn(64, device="cuda")
    result = ext.aten_stack_2_fwd(a, b)
    expected = aten.stack.default([a, b], 0)
    check("aten.stack", result, expected)
    print("PASS aten.stack")

if __name__ == "__main__":
    test()
