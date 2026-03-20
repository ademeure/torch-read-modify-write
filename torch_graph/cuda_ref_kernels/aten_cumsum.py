"""Reference CUDA kernel for aten.cumsum — cumulative sum along last dim."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

// One block per row, sequential scan within row
extern "C" __global__ void aten_cumsum_kernel(
    const float *input, float *output, unsigned int rows, unsigned int cols
) {
    unsigned int row = blockIdx.x;
    if (row >= rows) return;
    const float *ri = input + row * cols;
    float *ro = output + row * cols;
    float acc = 0.0f;
    for (unsigned int j = 0; j < cols; j++) {
        acc += ri[j];
        ro[j] = acc;
    }
}

torch::Tensor aten_cumsum_fwd(torch::Tensor input) {
    int cols = input.size(-1);
    int rows = input.numel() / cols;
    auto flat = input.reshape({rows, cols}).contiguous();
    auto output = torch::empty_like(flat);
    aten_cumsum_kernel<<<rows, 1>>>(
        flat.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
    return output.reshape(input.sizes());
}
"""

def test():
    ext = compile_cuda("aten_cumsum_kernel", KERNEL_SRC, ["aten_cumsum_fwd"])
    x = torch.randn(8, 64, device="cuda")
    result = ext.aten_cumsum_fwd(x)
    expected = aten.cumsum.default(x, -1)
    check("aten.cumsum", result, expected, atol=1e-4)
    print("PASS aten.cumsum")

if __name__ == "__main__":
    test()
