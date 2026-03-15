"""Reference CUDA kernel for aten.asin."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

KERNEL_SRC = r"""
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void aten_asin(const float *in0, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in0[i];
        out0[i] = asinf(x);
    }
}

torch::Tensor aten_asin_fwd(torch::Tensor in0) {
    auto out0 = torch::empty_like(in0);
    int n = in0.numel();
    aten_asin<<<(n+255)/256, 256>>>(in0.data_ptr<float>(), out0.data_ptr<float>(), n);
    return out0;
}
"""

def test():
    ext = compile_cuda("aten_asin", KERNEL_SRC, ["aten_asin_fwd"])
    x = torch.rand(1024, device='cuda') * 1.98 - 0.99
    result = ext.aten_asin_fwd(x)
    expected = x.asin()
    check("aten.asin", result, expected, atol=1e-05)
    print(f"PASS aten.asin")

if __name__ == "__main__":
    test()
