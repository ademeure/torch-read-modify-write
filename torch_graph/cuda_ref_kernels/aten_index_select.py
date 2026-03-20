"""Reference CUDA kernel for aten.index_select — select rows by index."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_index_select_kernel(
    const float *input, const long *index, float *output,
    unsigned int n_idx, unsigned int cols
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_idx * cols) return;
    unsigned int r = idx / cols, c = idx % cols;
    long src_r = index[r];
    output[idx] = input[src_r * cols + c];
}

torch::Tensor aten_index_select_fwd(torch::Tensor input, torch::Tensor index) {
    auto ci = input.contiguous();
    int cols = ci.size(1);
    int n_idx = index.numel();
    auto output = torch::empty({n_idx, cols}, ci.options());
    int total = n_idx * cols;
    aten_index_select_kernel<<<(total+255)/256, 256>>>(
        ci.data_ptr<float>(), index.data_ptr<long>(), output.data_ptr<float>(), n_idx, cols);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_index_select_kernel", KERNEL_SRC, ["aten_index_select_fwd"])
    x = torch.randn(32, 64, device="cuda")
    idx = torch.tensor([0, 5, 10, 15, 31], device="cuda")
    result = ext.aten_index_select_fwd(x, idx)
    expected = aten.index_select.default(x, 0, idx)
    check("aten.index_select", result, expected)
    print("PASS aten.index_select")

if __name__ == "__main__":
    test()
