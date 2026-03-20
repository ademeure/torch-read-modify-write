"""Reference CUDA kernel for aten._embedding_bag — bag of embeddings with sum reduction."""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_embedding_bag(
    const float *weight, const long *indices, const long *offsets,
    float *output, unsigned int num_bags, unsigned int embed_dim, unsigned int num_indices
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = num_bags * embed_dim;
    if (idx >= total) return;
    unsigned int d = idx % embed_dim;
    unsigned int bag = idx / embed_dim;
    unsigned int start = offsets[bag];
    unsigned int end = (bag + 1 < num_bags) ? offsets[bag + 1] : num_indices;
    float sum = 0.0f;
    for (unsigned int i = start; i < end; i++)
        sum += weight[indices[i] * embed_dim + d];
    output[idx] = sum;
}
"""

NUM_EMBED, EMBED_DIM, NUM_BAGS = 100, 32, 4

def init_once():
    weight = torch.randn(NUM_EMBED, EMBED_DIM, device="cuda")
    indices = torch.randint(0, NUM_EMBED, (16,), device="cuda")
    offsets = torch.tensor([0, 4, 8, 12], device="cuda")
    total = NUM_BAGS * EMBED_DIM
    result = torch.ops.aten._embedding_bag.default(weight, indices, offsets)
    return {"kernel_source": KERNEL_SRC, "inputs": [weight, indices, offsets],
            "expected": [result[0].flatten()],
            "outputs": ["float32;n=%d" % total], "grid": ((total + 255) // 256,), "atol": 1e-4}

def run(inputs, kernel):
    total = NUM_BAGS * EMBED_DIM
    return [kernel(*inputs, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.in_ptr(2), kernel.out_ptr(0),
        np.uint32(NUM_BAGS), np.uint32(EMBED_DIM), np.uint32(16),
    ])]
