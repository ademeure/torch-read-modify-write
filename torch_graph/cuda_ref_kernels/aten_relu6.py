"""Reference CUDA kernel for aten.relu6."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void aten_relu6(const float *in0, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in0[i];
        out0[i] = fminf(fmaxf(x, 0.0f), 6.0f);
    }
}

torch::Tensor aten_relu6_fwd(torch::Tensor in0) {
    auto out0 = torch::empty_like(in0);
    int n = in0.numel();
    aten_relu6<<<(n+255)/256, 256>>>(in0.data_ptr<float>(), out0.data_ptr<float>(), n);
    return out0;
}
"""

def test():
    ext = compile_cuda("aten_relu6", KERNEL_SRC, ["aten_relu6_fwd"])
    x = torch.randn(1024, device='cuda') * 5
    result = ext.aten_relu6_fwd(x)
    expected = aten.hardtanh.default(x, 0.0, 6.0)
    check("aten.relu6", result, expected, atol=1e-05)
    print(f"PASS aten.relu6")

if __name__ == "__main__":
    test()
