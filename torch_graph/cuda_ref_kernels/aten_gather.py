"""Reference CUDA kernel for aten.gather — gather along a dimension by index."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

// Gather along dim=1 for 2D tensors: out[i][j] = input[i][index[i][j]]
extern "C" __global__ void aten_gather_2d(
    const float *input, const long *index, float *output,
    unsigned int rows, unsigned int in_cols, unsigned int out_cols
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * out_cols) return;
    unsigned int r = idx / out_cols, c = idx % out_cols;
    long src_c = index[r * out_cols + c];
    output[idx] = input[r * in_cols + src_c];
}

torch::Tensor aten_gather_2d_fwd(torch::Tensor input, torch::Tensor index) {
    auto ci = input.contiguous();
    auto ci_idx = index.contiguous();
    int rows = ci.size(0), in_cols = ci.size(1), out_cols = ci_idx.size(1);
    auto output = torch::empty({rows, out_cols}, ci.options());
    int total = rows * out_cols;
    aten_gather_2d<<<(total+255)/256, 256>>>(
        ci.data_ptr<float>(), ci_idx.data_ptr<long>(), output.data_ptr<float>(),
        rows, in_cols, out_cols);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_gather_2d", KERNEL_SRC, ["aten_gather_2d_fwd"])
    x = torch.randn(8, 32, device="cuda")
    idx = torch.randint(0, 32, (8, 16), device="cuda")
    result = ext.aten_gather_2d_fwd(x, idx)
    expected = aten.gather.default(x, 1, idx)
    check("aten.gather", result, expected)
    print("PASS aten.gather")

if __name__ == "__main__":
    test()
