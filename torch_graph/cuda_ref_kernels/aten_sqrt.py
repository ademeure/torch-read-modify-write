"""Reference CUDA kernel for aten.sqrt."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void aten_sqrt(const float *in0, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in0[i];
        out0[i] = sqrtf(x);
    }
}

torch::Tensor aten_sqrt_fwd(torch::Tensor in0) {
    auto out0 = torch::empty_like(in0);
    int n = in0.numel();
    aten_sqrt<<<(n+255)/256, 256>>>(in0.data_ptr<float>(), out0.data_ptr<float>(), n);
    return out0;
}
"""

def test():
    ext = compile_cuda("aten_sqrt", KERNEL_SRC, ["aten_sqrt_fwd"])
    x = torch.rand(1024, device='cuda') + 0.01
    result = ext.aten_sqrt_fwd(x)
    expected = aten.sqrt.default(x)
    check("aten.sqrt", result, expected, atol=1e-06)
    print(f"PASS aten.sqrt")

if __name__ == "__main__":
    test()
