"""Reference CUDA kernel for aten.transpose — swap two dimensions, output contiguous."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

// 2D transpose: out[c][r] = in[r][c], output is contiguous
extern "C" __global__ void aten_transpose_2d(
    const float *in, float *out, unsigned int rows, unsigned int cols
) {
    unsigned int r = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows && c < cols) {
        out[c * rows + r] = in[r * cols + c];
    }
}

torch::Tensor aten_transpose_2d_fwd(torch::Tensor input) {
    auto ci = input.contiguous();
    int rows = ci.size(0), cols = ci.size(1);
    auto output = torch::empty({cols, rows}, ci.options());
    dim3 threads(16, 16);
    dim3 blocks((cols+15)/16, (rows+15)/16);
    aten_transpose_2d<<<blocks, threads>>>(
        ci.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_transpose_2d", KERNEL_SRC, ["aten_transpose_2d_fwd"])
    x = torch.randn(32, 64, device="cuda")
    result = ext.aten_transpose_2d_fwd(x)
    expected = aten.transpose.int(x, 0, 1).contiguous()
    check("aten.transpose", result, expected)
    print("PASS aten.transpose")

if __name__ == "__main__":
    test()
