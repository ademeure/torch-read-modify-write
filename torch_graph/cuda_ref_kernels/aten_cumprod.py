"""Reference CUDA kernel for aten.cumprod — cumulative product along last dim."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_cumprod_kernel(
    const float *input, float *output, unsigned int rows, unsigned int cols
) {
    unsigned int row = blockIdx.x;
    if (row >= rows) return;
    const float *ri = input + row * cols;
    float *ro = output + row * cols;
    float acc = 1.0f;
    for (unsigned int j = 0; j < cols; j++) {
        acc *= ri[j];
        ro[j] = acc;
    }
}

torch::Tensor aten_cumprod_fwd(torch::Tensor input) {
    int cols = input.size(-1);
    int rows = input.numel() / cols;
    auto flat = input.reshape({rows, cols}).contiguous();
    auto output = torch::empty_like(flat);
    aten_cumprod_kernel<<<rows, 1>>>(
        flat.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
    return output.reshape(input.sizes());
}
"""

def test():
    ext = compile_cuda("aten_cumprod_kernel", KERNEL_SRC, ["aten_cumprod_fwd"])
    x = torch.rand(8, 16, device="cuda") + 0.5
    result = ext.aten_cumprod_fwd(x)
    expected = aten.cumprod.default(x, -1)
    check("aten.cumprod", result, expected, atol=1e-3)
    print("PASS aten.cumprod")

if __name__ == "__main__":
    test()
