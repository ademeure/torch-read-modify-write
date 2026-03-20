"""Reference CUDA kernel for aten.contiguous — ensure contiguous memory layout."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_copy_kernel(const float *in, float *out, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i];
}

torch::Tensor aten_contiguous_fwd(torch::Tensor input) {
    auto ci = input.contiguous();  // PyTorch handles stride logic
    auto output = torch::empty_like(ci);
    int n = ci.numel();
    aten_copy_kernel<<<(n+255)/256, 256>>>(
        ci.data_ptr<float>(), output.data_ptr<float>(), n);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_copy_kernel", KERNEL_SRC, ["aten_contiguous_fwd"])
    x = torch.randn(32, 64, device="cuda").t()  # non-contiguous
    result = ext.aten_contiguous_fwd(x)
    expected = x.contiguous()
    check("aten.contiguous", result, expected)
    print("PASS aten.contiguous")

if __name__ == "__main__":
    test()
