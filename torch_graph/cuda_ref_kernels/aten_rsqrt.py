"""Reference CUDA kernel for aten.rsqrt."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void aten_rsqrt(const float *in0, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in0[i];
        out0[i] = rsqrtf(x);
    }
}

torch::Tensor aten_rsqrt_fwd(torch::Tensor in0) {
    auto out0 = torch::empty_like(in0);
    int n = in0.numel();
    aten_rsqrt<<<(n+255)/256, 256>>>(in0.data_ptr<float>(), out0.data_ptr<float>(), n);
    return out0;
}
"""

def test():
    ext = compile_cuda("aten_rsqrt", KERNEL_SRC, ["aten_rsqrt_fwd"])
    x = torch.rand(1024, device='cuda') + 0.01
    result = ext.aten_rsqrt_fwd(x)
    expected = aten.rsqrt.default(x)
    check("aten.rsqrt", result, expected, atol=0.0001)
    print(f"PASS aten.rsqrt")

if __name__ == "__main__":
    test()
