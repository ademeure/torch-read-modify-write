"""Reference CUDA kernel for aten.slice — copy a contiguous sub-range along a dim."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

// Slice along dim 0 of a 2D tensor: output = input[start:end:step, :]
extern "C" __global__ void aten_slice_2d(
    const float *input, float *output,
    unsigned int in_rows, unsigned int cols,
    unsigned int start, unsigned int step, unsigned int out_rows
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_rows * cols) return;
    unsigned int r = idx / cols;
    unsigned int c = idx % cols;
    unsigned int src_r = start + r * step;
    output[idx] = input[src_r * cols + c];
}

torch::Tensor aten_slice_2d_fwd(
    torch::Tensor input, int64_t start, int64_t end, int64_t step
) {
    auto ci = input.contiguous();
    int rows = ci.size(0), cols = ci.size(1);
    if (end > rows) end = rows;
    int out_rows = (end - start + step - 1) / step;
    auto output = torch::empty({out_rows, cols}, ci.options());
    int total = out_rows * cols;
    aten_slice_2d<<<(total+255)/256, 256>>>(
        ci.data_ptr<float>(), output.data_ptr<float>(), rows, cols, start, step, out_rows);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_slice_2d", KERNEL_SRC, ["aten_slice_2d_fwd"])
    x = torch.randn(32, 64, device="cuda")
    result = ext.aten_slice_2d_fwd(x, 4, 20, 1)
    expected = aten.slice.Tensor(x, 0, 4, 20).contiguous()
    check("aten.slice", result, expected)
    print("PASS aten.slice")

if __name__ == "__main__":
    test()
