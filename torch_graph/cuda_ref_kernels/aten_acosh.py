"""Reference CUDA kernel for aten.acosh."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

KERNEL_SRC = r"""
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void aten_acosh(const float *in0, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in0[i];
        out0[i] = acoshf(x);
    }
}

torch::Tensor aten_acosh_fwd(torch::Tensor in0) {
    auto out0 = torch::empty_like(in0);
    int n = in0.numel();
    aten_acosh<<<(n+255)/256, 256>>>(in0.data_ptr<float>(), out0.data_ptr<float>(), n);
    return out0;
}
"""

def test():
    ext = compile_cuda("aten_acosh", KERNEL_SRC, ["aten_acosh_fwd"])
    x = torch.rand(1024, device='cuda') + 1.01
    result = ext.aten_acosh_fwd(x)
    expected = x.acosh()
    check("aten.acosh", result, expected, atol=1e-05)
    print(f"PASS aten.acosh")

if __name__ == "__main__":
    test()
