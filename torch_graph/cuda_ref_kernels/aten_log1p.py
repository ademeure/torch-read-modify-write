"""Reference CUDA kernel for aten.log1p."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void aten_log1p(const float *in0, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in0[i];
        out0[i] = log1pf(x);
    }
}

torch::Tensor aten_log1p_fwd(torch::Tensor in0) {
    auto out0 = torch::empty_like(in0);
    int n = in0.numel();
    aten_log1p<<<(n+255)/256, 256>>>(in0.data_ptr<float>(), out0.data_ptr<float>(), n);
    return out0;
}
"""

def test():
    ext = compile_cuda("aten_log1p", KERNEL_SRC, ["aten_log1p_fwd"])
    x = torch.rand(1024, device='cuda')
    result = ext.aten_log1p_fwd(x)
    expected = aten.log1p.default(x)
    check("aten.log1p", result, expected, atol=1e-05)
    print(f"PASS aten.log1p")

if __name__ == "__main__":
    test()
