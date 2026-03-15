"""Reference CUDA kernel for aten.gelu."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

KERNEL_SRC = r"""
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void aten_gelu(const float *in0, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in0[i];
        out0[i] = (x * 0.5f * (1.0f + erff(x * 0.7071067811865476f)));
    }
}

torch::Tensor aten_gelu_fwd(torch::Tensor in0) {
    auto out0 = torch::empty_like(in0);
    int n = in0.numel();
    aten_gelu<<<(n+255)/256, 256>>>(in0.data_ptr<float>(), out0.data_ptr<float>(), n);
    return out0;
}
"""

def test():
    ext = compile_cuda("aten_gelu", KERNEL_SRC, ["aten_gelu_fwd"])
    x = torch.randn(1024, device='cuda')
    result = ext.aten_gelu_fwd(x)
    expected = torch.nn.functional.gelu(x)
    check("aten.gelu", result, expected, atol=1e-05)
    print(f"PASS aten.gelu")

if __name__ == "__main__":
    test()
