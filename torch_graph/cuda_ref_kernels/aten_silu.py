"""Reference CUDA kernel for aten.silu."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void aten_silu(const float *in0, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in0[i];
        out0[i] = (x / (1.0f + expf(-x)));
    }
}

torch::Tensor aten_silu_fwd(torch::Tensor in0) {
    auto out0 = torch::empty_like(in0);
    int n = in0.numel();
    aten_silu<<<(n+255)/256, 256>>>(in0.data_ptr<float>(), out0.data_ptr<float>(), n);
    return out0;
}
"""

def test():
    ext = compile_cuda("aten_silu", KERNEL_SRC, ["aten_silu_fwd"])
    x = torch.randn(1024, device='cuda')
    result = ext.aten_silu_fwd(x)
    expected = aten.silu.default(x)
    check("aten.silu", result, expected, atol=1e-05)
    print(f"PASS aten.silu")

if __name__ == "__main__":
    test()
