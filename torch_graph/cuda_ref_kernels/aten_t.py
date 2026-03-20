"""Reference CUDA kernel for aten.t — 2D matrix transpose (shorthand for transpose(0,1))."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_t_kernel(
    const float *in, float *out, unsigned int rows, unsigned int cols
) {
    unsigned int r = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows && c < cols)
        out[c * rows + r] = in[r * cols + c];
}

torch::Tensor aten_t_fwd(torch::Tensor input) {
    auto ci = input.contiguous();
    int rows = ci.size(0), cols = ci.size(1);
    auto output = torch::empty({cols, rows}, ci.options());
    dim3 threads(16, 16);
    dim3 blocks((cols+15)/16, (rows+15)/16);
    aten_t_kernel<<<blocks, threads>>>(
        ci.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_t_kernel", KERNEL_SRC, ["aten_t_fwd"])
    x = torch.randn(32, 64, device="cuda")
    result = ext.aten_t_fwd(x)
    expected = aten.t.default(x).contiguous()
    check("aten.t", result, expected)
    print("PASS aten.t")

if __name__ == "__main__":
    test()
