"""Reference CUDA kernel for aten.permute — 3D dimension permutation.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_permute.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_permute_3d(
    const float *input, float *output,
    unsigned int S0, unsigned int S1, unsigned int S2,
    unsigned int perm0, unsigned int perm1, unsigned int perm2
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = S0 * S1 * S2;
    if (idx >= total) return;
    unsigned int i0 = idx / (S1 * S2);
    unsigned int i1 = (idx / S2) % S1;
    unsigned int i2 = idx % S2;
    unsigned int in_idx[3];
    in_idx[0] = i0; in_idx[1] = i1; in_idx[2] = i2;
    unsigned int sizes[3];
    sizes[0] = S0; sizes[1] = S1; sizes[2] = S2;
    unsigned int perm[3];
    perm[0] = perm0; perm[1] = perm1; perm[2] = perm2;
    unsigned int out_sizes[3];
    out_sizes[0] = sizes[perm[0]]; out_sizes[1] = sizes[perm[1]]; out_sizes[2] = sizes[perm[2]];
    unsigned int o0 = in_idx[perm0], o1 = in_idx[perm1], o2 = in_idx[perm2];
    unsigned int out_idx = o0 * out_sizes[1] * out_sizes[2] + o1 * out_sizes[2] + o2;
    output[out_idx] = input[idx];
}
"""

S0, S1, S2 = 4, 8, 16

def init_once():
    x = torch.randn(S0, S1, S2, device="cuda")
    total = S0 * S1 * S2
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x],
        "expected": [torch.ops.aten.permute.default(x, [2, 0, 1]).contiguous().flatten()],
        "outputs": ["float32;n=%d" % total],
        "grid": ((total + 255) // 256,),
    }

def run(inputs, kernel):
    total = S0 * S1 * S2
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(S0), np.uint32(S1), np.uint32(S2),
        np.uint32(2), np.uint32(0), np.uint32(1),
    ])]
