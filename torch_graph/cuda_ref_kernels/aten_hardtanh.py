"""Reference CUDA kernel for aten.hardtanh."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void aten_hardtanh(const float *in0, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in0[i];
        out0[i] = fminf(fmaxf(x, -1.0f), 1.0f);
    }
}

torch::Tensor aten_hardtanh_fwd(torch::Tensor in0) {
    auto out0 = torch::empty_like(in0);
    int n = in0.numel();
    aten_hardtanh<<<(n+255)/256, 256>>>(in0.data_ptr<float>(), out0.data_ptr<float>(), n);
    return out0;
}
"""

def test():
    ext = compile_cuda("aten_hardtanh", KERNEL_SRC, ["aten_hardtanh_fwd"])
    x = torch.randn(1024, device='cuda') * 3
    result = ext.aten_hardtanh_fwd(x)
    expected = aten.hardtanh.default(x)
    check("aten.hardtanh", result, expected, atol=1e-05)
    print(f"PASS aten.hardtanh")

if __name__ == "__main__":
    test()
