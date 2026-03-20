"""Reference CUDA kernel for aten.hardswish."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void aten_hardswish(const float *in0, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in0[i];
        out0[i] = (x * fminf(fmaxf(x + 3.0f, 0.0f), 6.0f) / 6.0f);
    }
}

torch::Tensor aten_hardswish_fwd(torch::Tensor in0) {
    auto out0 = torch::empty_like(in0);
    int n = in0.numel();
    aten_hardswish<<<(n+255)/256, 256>>>(in0.data_ptr<float>(), out0.data_ptr<float>(), n);
    return out0;
}
"""

def test():
    ext = compile_cuda("aten_hardswish", KERNEL_SRC, ["aten_hardswish_fwd"])
    x = torch.randn(1024, device='cuda') * 5
    result = ext.aten_hardswish_fwd(x)
    expected = aten.hardswish.default(x)
    check("aten.hardswish", result, expected, atol=1e-05)
    print(f"PASS aten.hardswish")

if __name__ == "__main__":
    test()
