"""Reference CUDA kernel for aten.sign."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

KERNEL_SRC = r"""
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void aten_sign(const float *in0, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in0[i];
        out0[i] = ((x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f));
    }
}

torch::Tensor aten_sign_fwd(torch::Tensor in0) {
    auto out0 = torch::empty_like(in0);
    int n = in0.numel();
    aten_sign<<<(n+255)/256, 256>>>(in0.data_ptr<float>(), out0.data_ptr<float>(), n);
    return out0;
}
"""

def test():
    ext = compile_cuda("aten_sign", KERNEL_SRC, ["aten_sign_fwd"])
    x = torch.randn(1024, device='cuda')
    result = ext.aten_sign_fwd(x)
    expected = x.sign()
    check("aten.sign", result, expected)
    print(f"PASS aten.sign")

if __name__ == "__main__":
    test()
