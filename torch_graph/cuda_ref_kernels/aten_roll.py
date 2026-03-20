"""Reference CUDA kernel for aten.roll — circular shift along a dimension."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_roll_1d(
    const float *input, float *output, unsigned int n, int shift
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int src = ((int)i - shift % (int)n + (int)n) % (int)n;
    output[i] = input[src];
}

torch::Tensor aten_roll_1d_fwd(torch::Tensor input, int64_t shift) {
    auto ci = input.contiguous();
    int n = ci.numel();
    auto output = torch::empty_like(ci);
    aten_roll_1d<<<(n+255)/256, 256>>>(
        ci.data_ptr<float>(), output.data_ptr<float>(), n, (int)shift);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_roll_1d", KERNEL_SRC, ["aten_roll_1d_fwd"])
    x = torch.randn(256, device="cuda")
    result = ext.aten_roll_1d_fwd(x, 10)
    expected = aten.roll.default(x, [10]).contiguous()
    check("aten.roll", result, expected)
    print("PASS aten.roll")

if __name__ == "__main__":
    test()
