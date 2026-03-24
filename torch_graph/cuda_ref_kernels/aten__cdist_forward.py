"""Reference CUDA kernel for aten._cdist_forward — pairwise L2 distances."""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_cdist(
    const float *x1, const float *x2, float *out,
    unsigned int B, unsigned int M, unsigned int N, unsigned int D
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = B * M * N;
    if (idx >= total) return;
    unsigned int n = idx % N, m = (idx / N) % M, b = idx / (N * M);
    float sum = 0.0f;
    for (unsigned int d = 0; d < D; d++) {
        float diff = x1[b*M*D + m*D + d] - x2[b*N*D + n*D + d];
        sum += (diff != diff) ? __int_as_float(0x7FC00000) : diff * diff;
    }
    out[idx] = sqrtf(sum);
}
"""

# Self-registration metadata
DIMS = {"B": 2, "M": 8, "N": 6, "D": 4}
ATOL = 1e-4

def make_inputs(dims, seed):
    from torch_graph.cuda_ref_kernels._registry import _seeded
    return [_seeded((dims["B"], dims["M"], dims["D"]), seed),
            _seeded((dims["B"], dims["N"], dims["D"]), seed + 100)]

def reference(inputs):
    return [torch.ops.aten._cdist_forward.default(*inputs, 2.0, None).flatten()]

def init_once():
    inputs = make_inputs(DIMS, 1)
    total = DIMS["B"] * DIMS["M"] * DIMS["N"]
    return {"kernel_source": KERNEL_SRC, "inputs": inputs,
            "expected": reference(inputs),
            "outputs": ["float32;n=%d" % total],
            "grid": ((total + 255) // 256,), "atol": ATOL}

def run(inputs, kernel):
    total = DIMS["B"] * DIMS["M"] * DIMS["N"]
    return [kernel(*inputs, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.out_ptr(0),
        np.uint32(DIMS["B"]), np.uint32(DIMS["M"]),
        np.uint32(DIMS["N"]), np.uint32(DIMS["D"]),
    ])]
