"""Reference CUDA kernel for aten.triu — upper triangle of a matrix."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_triu_kernel(
    const float *input, float *output, unsigned int rows, unsigned int cols, int diagonal
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    unsigned int r = idx / cols, c = idx % cols;
    output[idx] = ((int)c >= (int)r + diagonal) ? input[idx] : 0.0f;
}

torch::Tensor aten_triu_fwd(torch::Tensor input, int64_t diagonal) {
    auto ci = input.contiguous();
    int rows = ci.size(0), cols = ci.size(1);
    auto output = torch::empty_like(ci);
    int total = rows * cols;
    aten_triu_kernel<<<(total+255)/256, 256>>>(
        ci.data_ptr<float>(), output.data_ptr<float>(), rows, cols, (int)diagonal);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_triu_kernel", KERNEL_SRC, ["aten_triu_fwd"])
    x = torch.randn(16, 16, device="cuda")
    result = ext.aten_triu_fwd(x, 0)
    expected = aten.triu.default(x)
    check("aten.triu", result, expected)
    print("PASS aten.triu")

if __name__ == "__main__":
    test()
