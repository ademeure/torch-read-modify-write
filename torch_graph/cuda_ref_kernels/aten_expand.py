"""Reference CUDA kernel for aten.expand — broadcast copy to larger shape."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

// 2D expand: input [1, N] or [M, 1] → output [M, N]
extern "C" __global__ void aten_expand_2d(
    const float *input, float *output,
    unsigned int in_rows, unsigned int in_cols,
    unsigned int out_rows, unsigned int out_cols
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_rows * out_cols) return;
    unsigned int r = idx / out_cols;
    unsigned int c = idx % out_cols;
    unsigned int ir = (in_rows == 1) ? 0 : r;
    unsigned int ic = (in_cols == 1) ? 0 : c;
    output[idx] = input[ir * in_cols + ic];
}

torch::Tensor aten_expand_2d_fwd(
    torch::Tensor input, int64_t out_rows, int64_t out_cols
) {
    auto ci = input.contiguous();
    int ir = ci.size(0), ic = ci.size(1);
    auto output = torch::empty({out_rows, out_cols}, ci.options());
    int total = out_rows * out_cols;
    aten_expand_2d<<<(total+255)/256, 256>>>(
        ci.data_ptr<float>(), output.data_ptr<float>(), ir, ic, out_rows, out_cols);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_expand_2d", KERNEL_SRC, ["aten_expand_2d_fwd"])
    x = torch.randn(1, 64, device="cuda")
    result = ext.aten_expand_2d_fwd(x, 32, 64)
    expected = aten.expand.default(x, [32, 64]).contiguous()
    check("aten.expand", result, expected)
    print("PASS aten.expand")

if __name__ == "__main__":
    test()
