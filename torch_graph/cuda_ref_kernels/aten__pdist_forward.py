"""Reference CUDA kernel for aten._pdist_forward — pairwise distances within one set."""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_pdist(
    const float *x, float *out, unsigned int N, unsigned int D, unsigned int num_pairs
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;
    // Map linear index to (i, j) pair where i < j
    // Using quadratic formula to find i
    unsigned int i = N - 1 - (unsigned int)floorf((-1.0f + sqrtf(1.0f + 8.0f*(float)(num_pairs - 1 - idx))) * 0.5f);
    unsigned int j = idx - (2*N - i - 1) * i / 2 + i + 1;
    if (j >= N) { i++; j = idx - (2*N - i - 1) * i / 2 + i + 1; }
    float sum = 0.0f;
    for (unsigned int d = 0; d < D; d++) {
        float diff = x[i*D + d] - x[j*D + d];
        sum += diff * diff;
    }
    out[idx] = sqrtf(sum);
}
"""

NN, DD = 8, 4
NUM_PAIRS = NN * (NN - 1) // 2

def init_once():
    x = torch.randn(NN, DD, device="cuda")
    return {"kernel_source": KERNEL_SRC, "inputs": [x.contiguous()],
            "expected": [torch.ops.aten._pdist_forward.default(x, 2.0).flatten()],
            "outputs": ["float32;n=%d" % NUM_PAIRS], "grid": ((NUM_PAIRS + 255) // 256,), "atol": 1e-4}

def run(inputs, kernel):
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(NN), np.uint32(DD), np.uint32(NUM_PAIRS),
    ])]
