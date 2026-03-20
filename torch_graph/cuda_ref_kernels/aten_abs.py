"""Reference CUDA kernel for aten.abs."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void aten_abs(const float *in0, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in0[i];
        out0[i] = fabsf(x);
    }
}

torch::Tensor aten_abs_fwd(torch::Tensor in0) {
    auto out0 = torch::empty_like(in0);
    int n = in0.numel();
    aten_abs<<<(n+255)/256, 256>>>(in0.data_ptr<float>(), out0.data_ptr<float>(), n);
    return out0;
}
"""

def test():
    ext = compile_cuda("aten_abs", KERNEL_SRC, ["aten_abs_fwd"])
    x = torch.randn(1024, device='cuda')
    result = ext.aten_abs_fwd(x)
    expected = aten.abs.default(x)
    check("aten.abs", result, expected)
    print(f"PASS aten.abs")

if __name__ == "__main__":
    test()
