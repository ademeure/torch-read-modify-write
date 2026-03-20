"""Reference CUDA kernel for aten.full — create tensor filled with a value."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_fill_val(float *output, float value, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = value;
}

torch::Tensor aten_full_fwd(int64_t d0, int64_t d1, double value) {
    auto output = torch::empty({d0, d1}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    int n = output.numel();
    aten_fill_val<<<(n+255)/256, 256>>>(output.data_ptr<float>(), (float)value, n);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_fill_val", KERNEL_SRC, ["aten_full_fwd"])
    result = ext.aten_full_fwd(32, 64, 3.14)
    expected = torch.full((32, 64), 3.14, device='cuda')
    check("aten.full", result, expected)
    print("PASS aten.full")

if __name__ == "__main__":
    test()
