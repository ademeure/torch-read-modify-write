"""Reference CUDA kernel for aten.fill — fill tensor with scalar value."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_fill_kernel(float *out, float value, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = value;
}

torch::Tensor aten_fill_fwd(torch::Tensor input, double value) {
    auto output = torch::empty_like(input.contiguous());
    int n = output.numel();
    aten_fill_kernel<<<(n+255)/256, 256>>>(output.data_ptr<float>(), (float)value, n);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_fill_kernel", KERNEL_SRC, ["aten_fill_fwd"])
    x = torch.randn(1024, device="cuda")
    result = ext.aten_fill_fwd(x, 3.14)
    expected = aten.fill.Scalar(x, 3.14)
    check("aten.fill", result, expected)
    print("PASS aten.fill")

if __name__ == "__main__":
    test()
