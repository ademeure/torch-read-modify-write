"""Reference CUDA kernel for aten.pow."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void aten_pow(const float *in0, const float *in1, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float a = in0[i];
        float b = in1[i];
        out0[i] = powf(a, b);
    }
}

torch::Tensor aten_pow_fwd(torch::Tensor in0, torch::Tensor in1) {
    auto out0 = torch::empty_like(in0);
    int n = in0.numel();
    aten_pow<<<(n+255)/256, 256>>>(in0.data_ptr<float>(), in1.data_ptr<float>(), out0.data_ptr<float>(), n);
    return out0;
}
"""

def test():
    ext = compile_cuda("aten_pow", KERNEL_SRC, ["aten_pow_fwd"])
    a = torch.rand(1024, device='cuda') + 0.1
    b = torch.rand(1024, device='cuda') * 3
    result = ext.aten_pow_fwd(a, b)
    expected = aten.pow.Tensor_Tensor(a, b)
    check("aten.pow", result, expected, atol=0.0001)
    print(f"PASS aten.pow")

if __name__ == "__main__":
    test()
