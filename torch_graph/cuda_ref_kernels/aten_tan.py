"""Reference CUDA kernel for aten.tan."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void aten_tan(const float *in0, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in0[i];
        out0[i] = tanf(x);
    }
}

torch::Tensor aten_tan_fwd(torch::Tensor in0) {
    auto out0 = torch::empty_like(in0);
    int n = in0.numel();
    aten_tan<<<(n+255)/256, 256>>>(in0.data_ptr<float>(), out0.data_ptr<float>(), n);
    return out0;
}
"""

def test():
    ext = compile_cuda("aten_tan", KERNEL_SRC, ["aten_tan_fwd"])
    x = torch.randn(1024, device='cuda') * 0.5
    result = ext.aten_tan_fwd(x)
    expected = aten.tan.default(x)
    check("aten.tan", result, expected, atol=0.0001)
    print(f"PASS aten.tan")

if __name__ == "__main__":
    test()
