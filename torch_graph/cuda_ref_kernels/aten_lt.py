"""Reference CUDA kernel for aten.lt."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_lt(const float *in0, const float *in1, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float a = in0[i];
        float b = in1[i];
        out0[i] = (a < b ? 1.0f : 0.0f);
    }
}

torch::Tensor aten_lt_fwd(torch::Tensor in0, torch::Tensor in1) {
    auto out0 = torch::empty_like(in0);
    int n = in0.numel();
    aten_lt<<<(n+255)/256, 256>>>(in0.data_ptr<float>(), in1.data_ptr<float>(), out0.data_ptr<float>(), n);
    return out0;
}
"""

def test():
    ext = compile_cuda("aten_lt", KERNEL_SRC, ["aten_lt_fwd"])
    a = torch.randn(1024, device='cuda')
    b = torch.randn(1024, device='cuda')
    result = ext.aten_lt_fwd(a, b)
    expected = aten.lt.Tensor(a, b).float()
    check("aten.lt", result, expected)
    print(f"PASS aten.lt")

if __name__ == "__main__":
    test()
