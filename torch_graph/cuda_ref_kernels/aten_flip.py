"""Reference CUDA kernel for aten.flip — reverse along given dimensions."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

// Flip last dim of 2D tensor
extern "C" __global__ void aten_flip_2d(
    const float *input, float *output, unsigned int rows, unsigned int cols
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    unsigned int r = idx / cols, c = idx % cols;
    output[r * cols + (cols - 1 - c)] = input[idx];
}

torch::Tensor aten_flip_2d_fwd(torch::Tensor input) {
    auto ci = input.contiguous();
    int rows = ci.size(0), cols = ci.size(1);
    auto output = torch::empty_like(ci);
    int total = rows * cols;
    aten_flip_2d<<<(total+255)/256, 256>>>(
        ci.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_flip_2d", KERNEL_SRC, ["aten_flip_2d_fwd"])
    x = torch.randn(16, 32, device="cuda")
    result = ext.aten_flip_2d_fwd(x)
    expected = aten.flip.default(x, [-1]).contiguous()
    check("aten.flip", result, expected)
    print("PASS aten.flip")

if __name__ == "__main__":
    test()
