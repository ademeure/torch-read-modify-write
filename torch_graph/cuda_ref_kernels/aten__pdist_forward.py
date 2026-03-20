"""Reference CUDA kernel for aten._pdist_forward."""
import torch, numpy as np
KERNEL_SRC = r"""
extern "C" __global__ void aten_pdist(
    const float *x, float *out, unsigned int N, unsigned int D
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int num_pairs = N * (N - 1) / 2;
    if (idx >= num_pairs) return;
    // Map linear index to (i, j) pair
    unsigned int i = 0, remaining = idx;
    for (i = 0; i < N; i++) {
        unsigned int row_size = N - 1 - i;
        if (remaining < row_size) break;
        remaining -= row_size;
    }
    unsigned int j = i + 1 + remaining;
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
        kernel.in_ptr(0), kernel.out_ptr(0), np.uint32(NN), np.uint32(DD)])]
