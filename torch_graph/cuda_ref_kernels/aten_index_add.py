"""Reference CUDA kernel for aten.index_add — add source into self at indices."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_index_add_init(const float *self, float *out, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = self[i];
}

extern "C" __global__ void aten_index_add_kernel(
    const long *index, const float *source, float *out,
    unsigned int n_idx, unsigned int cols
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_idx * cols) return;
    unsigned int r = idx / cols, c = idx % cols;
    long dst_r = index[r];
    atomicAdd(&out[dst_r * cols + c], source[idx]);
}

torch::Tensor aten_index_add_fwd(torch::Tensor self, torch::Tensor index, torch::Tensor source) {
    auto ci = self.contiguous();
    int rows = ci.size(0), cols = ci.size(1), n_idx = index.numel();
    auto out = torch::empty_like(ci);
    int n = ci.numel();
    aten_index_add_init<<<(n+255)/256, 256>>>(ci.data_ptr<float>(), out.data_ptr<float>(), n);
    int total = n_idx * cols;
    aten_index_add_kernel<<<(total+255)/256, 256>>>(
        index.data_ptr<long>(), source.data_ptr<float>(), out.data_ptr<float>(), n_idx, cols);
    return out;
}
"""

def test():
    ext = compile_cuda("aten_index_add_init", KERNEL_SRC, ["aten_index_add_fwd"])
    x = torch.zeros(32, 64, device="cuda")
    idx = torch.tensor([0, 5, 10, 15, 20], device="cuda")
    src = torch.randn(5, 64, device="cuda")
    result = ext.aten_index_add_fwd(x, idx, src)
    expected = aten.index_add.default(x, 0, idx, src)
    check("aten.index_add", result, expected, atol=1e-5)
    print("PASS aten.index_add")

if __name__ == "__main__":
    test()
