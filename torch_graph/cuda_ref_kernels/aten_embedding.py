"""Reference CUDA kernel for aten.embedding — table lookup."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_embedding_kernel(
    const float *weight, const long *indices, float *output,
    unsigned int n_idx, unsigned int embed_dim
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_idx * embed_dim) return;
    unsigned int r = idx / embed_dim, c = idx % embed_dim;
    long row = indices[r];
    output[idx] = weight[row * embed_dim + c];
}

torch::Tensor aten_embedding_fwd(torch::Tensor weight, torch::Tensor indices) {
    int n_idx = indices.numel();
    int embed_dim = weight.size(1);
    auto output = torch::empty({n_idx, embed_dim}, weight.options());
    int total = n_idx * embed_dim;
    aten_embedding_kernel<<<(total+255)/256, 256>>>(
        weight.data_ptr<float>(), indices.data_ptr<long>(),
        output.data_ptr<float>(), n_idx, embed_dim);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_embedding_kernel", KERNEL_SRC, ["aten_embedding_fwd"])
    weight = torch.randn(100, 64, device="cuda")
    indices = torch.randint(0, 100, (32,), device="cuda")
    result = ext.aten_embedding_fwd(weight, indices)
    expected = aten.embedding.default(weight, indices)
    check("aten.embedding", result, expected)
    print("PASS aten.embedding")

if __name__ == "__main__":
    test()
