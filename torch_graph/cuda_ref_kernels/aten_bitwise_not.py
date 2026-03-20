"""Reference CUDA kernel for aten.bitwise_not."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void aten_bitwise_not(const float *in0, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in0[i];
        out0[i] = ((x == 0.0f) ? 1.0f : 0.0f);
    }
}

torch::Tensor aten_bitwise_not_fwd(torch::Tensor in0) {
    auto out0 = torch::empty_like(in0);
    int n = in0.numel();
    aten_bitwise_not<<<(n+255)/256, 256>>>(in0.data_ptr<float>(), out0.data_ptr<float>(), n);
    return out0;
}
"""

def test():
    ext = compile_cuda("aten_bitwise_not", KERNEL_SRC, ["aten_bitwise_not_fwd"])
    x = torch.tensor([1.0, 0.0, -1.0, 0.0, 3.14] * 200, device='cuda')
    result = ext.aten_bitwise_not_fwd(x)
    expected = aten.logical_not.default(x).float()
    check("aten.bitwise_not", result, expected)
    print(f"PASS aten.bitwise_not")

if __name__ == "__main__":
    test()
