"""Reference CUDA kernel for aten.select — select a single index along a dim."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

// Select index along dim 0 of a 2D tensor: output = input[index, :]
extern "C" __global__ void aten_select_copy(
    const float *input, float *output, unsigned int cols, unsigned int index
) {
    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < cols)
        output[c] = input[index * cols + c];
}

torch::Tensor aten_select_fwd(torch::Tensor input, int64_t index) {
    auto ci = input.contiguous();
    int cols = ci.size(1);
    auto output = torch::empty({cols}, ci.options());
    aten_select_copy<<<(cols+255)/256, 256>>>(
        ci.data_ptr<float>(), output.data_ptr<float>(), cols, index);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_select_copy", KERNEL_SRC, ["aten_select_fwd"])
    x = torch.randn(32, 64, device="cuda")
    result = ext.aten_select_fwd(x, 5)
    expected = aten.select.int(x, 0, 5).contiguous()
    check("aten.select", result, expected)
    print("PASS aten.select")

if __name__ == "__main__":
    test()
