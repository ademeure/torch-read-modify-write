"""Reference CUDA kernel for aten.atan."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void aten_atan(const float *in0, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in0[i];
        out0[i] = atanf(x);
    }
}

torch::Tensor aten_atan_fwd(torch::Tensor in0) {
    auto out0 = torch::empty_like(in0);
    int n = in0.numel();
    aten_atan<<<(n+255)/256, 256>>>(in0.data_ptr<float>(), out0.data_ptr<float>(), n);
    return out0;
}
"""

def test():
    ext = compile_cuda("aten_atan", KERNEL_SRC, ["aten_atan_fwd"])
    x = torch.randn(1024, device='cuda')
    result = ext.aten_atan_fwd(x)
    expected = aten.atan.default(x)
    check("aten.atan", result, expected, atol=1e-05)
    print(f"PASS aten.atan")

if __name__ == "__main__":
    test()
