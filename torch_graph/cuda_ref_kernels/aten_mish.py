"""Reference CUDA kernel for aten.mish."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void aten_mish(const float *in0, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in0[i];
        out0[i] = (x * tanhf((x > 20.0f) ? x : logf(1.0f + expf(x))));
    }
}

torch::Tensor aten_mish_fwd(torch::Tensor in0) {
    auto out0 = torch::empty_like(in0);
    int n = in0.numel();
    aten_mish<<<(n+255)/256, 256>>>(in0.data_ptr<float>(), out0.data_ptr<float>(), n);
    return out0;
}
"""

def test():
    ext = compile_cuda("aten_mish", KERNEL_SRC, ["aten_mish_fwd"])
    x = torch.randn(1024, device='cuda')
    result = ext.aten_mish_fwd(x)
    expected = aten.mish.default(x)
    check("aten.mish", result, expected, atol=0.0001)
    print(f"PASS aten.mish")

if __name__ == "__main__":
    test()
