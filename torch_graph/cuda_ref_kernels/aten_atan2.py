"""Reference CUDA kernel for aten.atan2."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void aten_atan2(const float *in0, const float *in1, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float a = in0[i];
        float b = in1[i];
        out0[i] = atan2f(a, b);
    }
}

torch::Tensor aten_atan2_fwd(torch::Tensor in0, torch::Tensor in1) {
    auto out0 = torch::empty_like(in0);
    int n = in0.numel();
    aten_atan2<<<(n+255)/256, 256>>>(in0.data_ptr<float>(), in1.data_ptr<float>(), out0.data_ptr<float>(), n);
    return out0;
}
"""

def test():
    ext = compile_cuda("aten_atan2", KERNEL_SRC, ["aten_atan2_fwd"])
    a = torch.randn(1024, device='cuda')
    b = torch.randn(1024, device='cuda')
    result = ext.aten_atan2_fwd(a, b)
    expected = aten.atan2.default(a, b)
    check("aten.atan2", result, expected, atol=1e-05)
    print(f"PASS aten.atan2")

if __name__ == "__main__":
    test()
