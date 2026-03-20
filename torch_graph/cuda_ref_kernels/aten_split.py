"""Reference CUDA kernel for aten.split — split tensor into chunks along dim."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_split_copy(
    const float *input, float *out, unsigned int offset, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = input[offset + i];
}

std::vector<torch::Tensor> aten_split_fwd(
    torch::Tensor input, int64_t chunk_size, int64_t dim
) {
    auto ci = input.contiguous();
    int total = ci.size(dim);
    std::vector<torch::Tensor> result;
    int cols = ci.numel() / ci.size(0);
    for (int start = 0; start < total; start += chunk_size) {
        int len = std::min((int)chunk_size, total - start);
        int n = len * cols;
        auto out = torch::empty({len, cols}, ci.options());
        aten_split_copy<<<(n+255)/256, 256>>>(
            ci.data_ptr<float>(), out.data_ptr<float>(), start * cols, n);
        result.push_back(out);
    }
    return result;
}
"""

def test():
    ext = compile_cuda("aten_split_copy", KERNEL_SRC, ["aten_split_fwd"])
    x = torch.randn(32, 64, device="cuda")
    result = ext.aten_split_fwd(x, 8, 0)
    expected = aten.split.Tensor(x, 8, 0)
    for i, (r, e) in enumerate(zip(result, expected)):
        check(f"aten.split[{i}]", r, e.contiguous())
    print("PASS aten.split")

if __name__ == "__main__":
    test()
