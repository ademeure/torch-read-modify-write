"""Reference CUDA kernel for aten.remainder."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

KERNEL_SRC = r"""
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void aten_remainder(const float *in0, const float *in1, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float a = in0[i];
        float b = in1[i];
        out0[i] = (a - b * floorf(a / b));
    }
}

torch::Tensor aten_remainder_fwd(torch::Tensor in0, torch::Tensor in1) {
    auto out0 = torch::empty_like(in0);
    int n = in0.numel();
    aten_remainder<<<(n+255)/256, 256>>>(in0.data_ptr<float>(), in1.data_ptr<float>(), out0.data_ptr<float>(), n);
    return out0;
}
"""

def test():
    ext = compile_cuda("aten_remainder", KERNEL_SRC, ["aten_remainder_fwd"])
    a = torch.randn(1024, device='cuda') * 10
    b = torch.randn(1024, device='cuda').abs() + 0.5
    result = ext.aten_remainder_fwd(a, b)
    expected = torch.remainder(a, b)
    check("aten.remainder", result, expected, atol=0.0001)
    print(f"PASS aten.remainder")

if __name__ == "__main__":
    test()
