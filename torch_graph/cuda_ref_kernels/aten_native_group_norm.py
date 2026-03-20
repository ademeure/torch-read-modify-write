"""Reference CUDA kernel for aten.native_group_norm — group normalization."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void aten_group_norm(
    const float *input, const float *weight, const float *bias,
    float *output, unsigned int N, unsigned int C, unsigned int HW,
    unsigned int G, float eps
) {
    // One block per (n, g) pair
    unsigned int ng = blockIdx.x;
    unsigned int n = ng / G, g = ng % G;
    unsigned int tid = threadIdx.x;
    unsigned int CpG = C / G;  // channels per group
    unsigned int group_size = CpG * HW;

    extern __shared__ float sdata[];
    float *s_sum = sdata, *s_sq = sdata + blockDim.x;

    // Compute mean and var for this group
    float ls = 0.0f, lsq = 0.0f;
    for (unsigned int i = tid; i < group_size; i += blockDim.x) {
        unsigned int c_local = i / HW;
        unsigned int hw = i % HW;
        unsigned int c = g * CpG + c_local;
        float v = input[n * C * HW + c * HW + hw];
        ls += v; lsq += v * v;
    }
    s_sum[tid] = ls; s_sq[tid] = lsq;
    __syncthreads();
    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) { s_sum[tid] += s_sum[tid+s]; s_sq[tid] += s_sq[tid+s]; }
        __syncthreads();
    }
    float mean = s_sum[0] / (float)group_size;
    float rstd = rsqrtf(s_sq[0] / (float)group_size - mean * mean + eps);

    // Normalize
    for (unsigned int i = tid; i < group_size; i += blockDim.x) {
        unsigned int c_local = i / HW;
        unsigned int hw = i % HW;
        unsigned int c = g * CpG + c_local;
        unsigned int idx = n * C * HW + c * HW + hw;
        output[idx] = (input[idx] - mean) * rstd * weight[c] + bias[c];
    }
}

torch::Tensor aten_group_norm_fwd(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int64_t N, int64_t C, int64_t HW, int64_t G, double eps
) {
    auto output = torch::empty_like(input);
    int threads = 256;
    int smem = 2 * threads * sizeof(float);
    aten_group_norm<<<N * G, threads, smem>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), N, C, HW, G, (float)eps);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_group_norm", KERNEL_SRC, ["aten_group_norm_fwd"])
    x = torch.randn(2, 8, 4, 4, device="cuda")
    w = torch.randn(8, device="cuda")
    b = torch.randn(8, device="cuda")
    result = ext.aten_group_norm_fwd(x.contiguous(), w, b, 2, 8, 16, 4, 1e-5)
    expected = aten.native_group_norm.default(x, w, b, 2, 8, 16, 4, 1e-5)
    check("aten.native_group_norm", result, expected[0], atol=1e-4)
    print("PASS aten.native_group_norm")

if __name__ == "__main__":
    test()
