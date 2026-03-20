"""Reference CUDA kernel for aten.narrow — narrow view along a dimension."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_narrow_copy(
    const float *input, float *output, unsigned int cols,
    unsigned int start, unsigned int length
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= length * cols) return;
    unsigned int r = idx / cols, c = idx % cols;
    output[idx] = input[(start + r) * cols + c];
}

torch::Tensor aten_narrow_fwd(torch::Tensor input, int64_t start, int64_t length) {
    auto ci = input.contiguous();
    int cols = ci.size(1);
    auto output = torch::empty({length, cols}, ci.options());
    int total = length * cols;
    aten_narrow_copy<<<(total+255)/256, 256>>>(
        ci.data_ptr<float>(), output.data_ptr<float>(), cols, start, length);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_narrow_copy", KERNEL_SRC, ["aten_narrow_fwd"])
    x = torch.randn(32, 64, device="cuda")
    result = ext.aten_narrow_fwd(x, 4, 10)
    expected = aten.narrow.default(x, 0, 4, 10).contiguous()
    check("aten.narrow", result, expected)
    print("PASS aten.narrow")

if __name__ == "__main__":
    test()
