"""Reference CUDA kernel for aten.native_layer_norm."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void aten_layer_norm(
    const float *input, const float *weight, const float *bias,
    float *output, float *mean_out, float *rstd_out,
    unsigned int rows, unsigned int cols, float eps
) {
    extern __shared__ float sdata[];
    float *s_sum = sdata, *s_sq = sdata + blockDim.x;
    unsigned int row = blockIdx.x, tid = threadIdx.x;
    if (row >= rows) return;
    const float *ri = input + row * cols;
    float *ro = output + row * cols;
    float ls = 0.0f, lsq = 0.0f;
    for (unsigned int j = tid; j < cols; j += blockDim.x) {
        float v = ri[j]; ls += v; lsq += v * v;
    }
    s_sum[tid] = ls; s_sq[tid] = lsq; __syncthreads();
    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) { s_sum[tid] += s_sum[tid+s]; s_sq[tid] += s_sq[tid+s]; }
        __syncthreads();
    }
    float mean = s_sum[0] / (float)cols;
    float rstd = rsqrtf(s_sq[0] / (float)cols - mean * mean + eps);
    if (tid == 0) { mean_out[row] = mean; rstd_out[row] = rstd; }
    __syncthreads();
    for (unsigned int j = tid; j < cols; j += blockDim.x)
        ro[j] = (ri[j] - mean) * rstd * weight[j] + bias[j];
}

std::vector<torch::Tensor> aten_layer_norm_fwd(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, double eps
) {
    int cols = input.size(-1);
    int rows = input.numel() / cols;
    auto flat = input.reshape({rows, cols}).contiguous();
    auto output = torch::empty_like(flat);
    auto mean_out = torch::empty({rows}, input.options());
    auto rstd_out = torch::empty({rows}, input.options());
    int threads = 256;
    int smem = 2 * threads * sizeof(float);
    aten_layer_norm<<<rows, threads, smem>>>(
        flat.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), mean_out.data_ptr<float>(), rstd_out.data_ptr<float>(),
        rows, cols, (float)eps);
    return {output.reshape(input.sizes()), mean_out.reshape({rows, 1}), rstd_out.reshape({rows, 1})};
}
"""

def test():
    ext = compile_cuda("aten_layer_norm", KERNEL_SRC, ["aten_layer_norm_fwd"])
    x = torch.randn(8, 64, device="cuda")
    w = torch.randn(64, device="cuda")
    b = torch.randn(64, device="cuda")
    result = ext.aten_layer_norm_fwd(x, w, b, 1e-5)
    expected = aten.native_layer_norm.default(x, [64], w, b, 1e-5)
    check("aten.native_layer_norm", result[0], expected[0], atol=1e-4)
    print("PASS aten.native_layer_norm")

if __name__ == "__main__":
    test()
