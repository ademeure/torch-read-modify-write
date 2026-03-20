"""Reference CUDA kernel for aten.hardsigmoid."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void aten_hardsigmoid(const float *in0, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in0[i];
        out0[i] = fminf(fmaxf(x / 6.0f + 0.5f, 0.0f), 1.0f);
    }
}

torch::Tensor aten_hardsigmoid_fwd(torch::Tensor in0) {
    auto out0 = torch::empty_like(in0);
    int n = in0.numel();
    aten_hardsigmoid<<<(n+255)/256, 256>>>(in0.data_ptr<float>(), out0.data_ptr<float>(), n);
    return out0;
}
"""

def test():
    ext = compile_cuda("aten_hardsigmoid", KERNEL_SRC, ["aten_hardsigmoid_fwd"])
    x = torch.randn(1024, device='cuda') * 5
    result = ext.aten_hardsigmoid_fwd(x)
    expected = aten.hardsigmoid.default(x)
    check("aten.hardsigmoid", result, expected, atol=1e-05)
    print(f"PASS aten.hardsigmoid")

if __name__ == "__main__":
    test()
