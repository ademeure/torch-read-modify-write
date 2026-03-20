"""Reference CUDA kernel for aten.embedding — table lookup.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_embedding.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
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
"""

def init_once():
    weight = torch.randn(100, 64, device="cuda")
    indices = torch.randint(0, 100, (32,), device="cuda")
    total = 32 * 64
    return {
        "kernel_source": KERNEL_SRC, "inputs": [weight, indices],
        "expected": [torch.ops.aten.embedding.default(weight, indices).flatten()],
        "outputs": ["float32;n=%d" % total],
        "grid": ((total + 255) // 256,),
    }

def run(inputs, kernel):
    weight, indices = inputs
    n_idx = indices.numel()
    embed_dim = weight.shape[1]
    return [kernel(weight, indices, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.out_ptr(0),
        np.uint32(n_idx), np.uint32(embed_dim),
    ])]
