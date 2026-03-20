"""Reference CUDA kernel for aten.zeros — create zero-filled tensor."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_fill_zero(float *output, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = 0.0f;
}

torch::Tensor aten_zeros_fwd(int64_t d0, int64_t d1) {
    auto output = torch::empty({d0, d1}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    int n = output.numel();
    aten_fill_zero<<<(n+255)/256, 256>>>(output.data_ptr<float>(), n);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_fill_zero", KERNEL_SRC, ["aten_zeros_fwd"])
    result = ext.aten_zeros_fwd(32, 64)
    expected = torch.zeros(32, 64, device='cuda')
    check("aten.zeros", result, expected)
    print("PASS aten.zeros")

if __name__ == "__main__":
    test()
