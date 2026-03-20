"""Reference CUDA kernel for aten.argmax — index of maximum value."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_argmax_kernel(
    const float *input, long *output, unsigned int rows, unsigned int cols
) {
    unsigned int row = blockIdx.x;
    if (row >= rows) return;
    const float *ri = input + row * cols;
    float best = ri[0];
    long best_idx = 0;
    for (unsigned int j = 1; j < cols; j++) {
        if (ri[j] > best) { best = ri[j]; best_idx = j; }
    }
    output[row] = best_idx;
}

torch::Tensor aten_argmax_fwd(torch::Tensor input) {
    int cols = input.size(-1);
    int rows = input.numel() / cols;
    auto flat = input.reshape({rows, cols}).contiguous();
    auto output = torch::empty({rows}, torch::TensorOptions().dtype(torch::kLong).device(input.device()));
    aten_argmax_kernel<<<rows, 1>>>(flat.data_ptr<float>(), output.data_ptr<long>(), rows, cols);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_argmax_kernel", KERNEL_SRC, ["aten_argmax_fwd"])
    x = torch.randn(32, 64, device="cuda")
    result = ext.aten_argmax_fwd(x)
    expected = aten.argmax.default(x, -1)
    check("aten.argmax", result, expected)
    print("PASS aten.argmax")

if __name__ == "__main__":
    test()
