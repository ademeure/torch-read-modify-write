"""Reference CUDA kernel for aten.clamp — clamp to [min, max] range."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_clamp(
    const float *in0, float *out0, float lo, float hi, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in0[i];
        out0[i] = fminf(fmaxf(x, lo), hi);
    }
}

torch::Tensor aten_clamp_fwd(torch::Tensor in0, double lo, double hi) {
    auto out0 = torch::empty_like(in0);
    int n = in0.numel();
    aten_clamp<<<(n+255)/256, 256>>>(
        in0.data_ptr<float>(), out0.data_ptr<float>(), (float)lo, (float)hi, n);
    return out0;
}
"""

def test():
    ext = compile_cuda("aten_clamp", KERNEL_SRC, ["aten_clamp_fwd"])
    x = torch.randn(1024, device='cuda') * 5
    result = ext.aten_clamp_fwd(x, -1.0, 1.0)
    expected = aten.clamp.default(x, -1.0, 1.0)
    check("aten.clamp", result, expected)
    print("PASS aten.clamp")

if __name__ == "__main__":
    test()
