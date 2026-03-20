"""Reference CUDA kernel for aten.permute — general dimension permutation, output contiguous."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

// 3D permute: input[d0][d1][d2] → output[dims[0]][dims[1]][dims[2]], contiguous
extern "C" __global__ void aten_permute_3d(
    const float *input, float *output,
    unsigned int S0, unsigned int S1, unsigned int S2,
    unsigned int perm0, unsigned int perm1, unsigned int perm2
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = S0 * S1 * S2;
    if (idx >= total) return;

    // Input indices
    unsigned int i0 = idx / (S1 * S2);
    unsigned int i1 = (idx / S2) % S1;
    unsigned int i2 = idx % S2;

    // Permuted shape sizes
    unsigned int in_idx[3] = {i0, i1, i2};
    unsigned int sizes[3] = {S0, S1, S2};
    unsigned int perm[3] = {perm0, perm1, perm2};

    // Output index (contiguous in permuted layout)
    unsigned int out_sizes[3] = {sizes[perm[0]], sizes[perm[1]], sizes[perm[2]]};
    unsigned int o0 = in_idx[perm0], o1 = in_idx[perm1], o2 = in_idx[perm2];
    // Wait — we need to map from OUTPUT index to INPUT index.
    // Easier: iterate over output and read from input.
    // But this kernel iterates over input. Let's compute where this input element goes.
    unsigned int out_idx = o0 * out_sizes[1] * out_sizes[2] + o1 * out_sizes[2] + o2;
    output[out_idx] = input[idx];
}

torch::Tensor aten_permute_3d_fwd(
    torch::Tensor input, int64_t p0, int64_t p1, int64_t p2
) {
    auto ci = input.contiguous();
    int S0 = ci.size(0), S1 = ci.size(1), S2 = ci.size(2);
    int out_sizes[3];
    int perm[3] = {(int)p0, (int)p1, (int)p2};
    int sizes[3] = {S0, S1, S2};
    for (int i = 0; i < 3; i++) out_sizes[i] = sizes[perm[i]];
    auto output = torch::empty({out_sizes[0], out_sizes[1], out_sizes[2]}, ci.options());
    int total = S0 * S1 * S2;
    aten_permute_3d<<<(total+255)/256, 256>>>(
        ci.data_ptr<float>(), output.data_ptr<float>(), S0, S1, S2, p0, p1, p2);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_permute_3d", KERNEL_SRC, ["aten_permute_3d_fwd"])
    x = torch.randn(4, 8, 16, device="cuda")
    result = ext.aten_permute_3d_fwd(x, 2, 0, 1)
    expected = aten.permute.default(x, [2, 0, 1]).contiguous()
    check("aten.permute", result, expected)
    print("PASS aten.permute")

if __name__ == "__main__":
    test()
