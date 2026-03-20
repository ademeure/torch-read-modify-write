"""Reference CUDA kernel for aten._to_copy — dtype/device conversion (copy)."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_to_copy_kernel(const float *in, float *out, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i];
}

torch::Tensor aten_to_copy_fwd(torch::Tensor input) {
    auto ci = input.contiguous();
    auto output = torch::empty_like(ci);
    int n = ci.numel();
    aten_to_copy_kernel<<<(n+255)/256, 256>>>(
        ci.data_ptr<float>(), output.data_ptr<float>(), n);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_to_copy_kernel", KERNEL_SRC, ["aten_to_copy_fwd"])
    x = torch.randn(1024, device="cuda")
    result = ext.aten_to_copy_fwd(x)
    expected = aten._to_copy.default(x)
    check("aten._to_copy", result, expected)
    print("PASS aten._to_copy")

if __name__ == "__main__":
    test()
