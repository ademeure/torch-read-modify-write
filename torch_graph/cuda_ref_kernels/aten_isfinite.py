"""Reference CUDA kernel for aten.isfinite."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void aten_isfinite(const float *in0, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in0[i];
        out0[i] = ((isfinite(x)) ? 1.0f : 0.0f);
    }
}

torch::Tensor aten_isfinite_fwd(torch::Tensor in0) {
    auto out0 = torch::empty_like(in0);
    int n = in0.numel();
    aten_isfinite<<<(n+255)/256, 256>>>(in0.data_ptr<float>(), out0.data_ptr<float>(), n);
    return out0;
}
"""

def test():
    ext = compile_cuda("aten_isfinite", KERNEL_SRC, ["aten_isfinite_fwd"])
    x = torch.tensor([1.0, float('inf'), 0.0, float('nan'), -1.0] * 200, device='cuda')
    result = ext.aten_isfinite_fwd(x)
    expected = torch.isfinite(x).float()
    check("aten.isfinite", result, expected)
    print(f"PASS aten.isfinite")

if __name__ == "__main__":
    test()
