"""Reference CUDA kernel for aten.var — variance reduction."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void aten_var_kernel(
    const float *input, float *output, unsigned int rows, unsigned int cols, int correction
) {
    extern __shared__ float sdata[];
    float *s_sum = sdata, *s_sq = sdata + blockDim.x;
    unsigned int row = blockIdx.x, tid = threadIdx.x;
    if (row >= rows) return;
    const float *ri = input + row * cols;
    float ls = 0.0f, lsq = 0.0f;
    for (unsigned int j = tid; j < cols; j += blockDim.x) {
        float v = ri[j]; ls += v; lsq += v * v;
    }
    s_sum[tid] = ls; s_sq[tid] = lsq; __syncthreads();
    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) { s_sum[tid] += s_sum[tid+s]; s_sq[tid] += s_sq[tid+s]; }
        __syncthreads();
    }
    if (tid == 0) {
        float mean = s_sum[0] / (float)cols;
        output[row] = (s_sq[0] / (float)cols - mean * mean) * (float)cols / (float)(cols - correction);
    }
}

torch::Tensor aten_var_fwd(torch::Tensor input) {
    auto sizes = input.sizes();
    int cols = sizes[sizes.size() - 1];
    int rows = input.numel() / cols;
    auto flat = input.reshape({rows, cols}).contiguous();
    auto output = torch::empty({rows}, input.options());
    int threads = 256;
    int smem = 2 * threads * sizeof(float);
    aten_var_kernel<<<rows, threads, smem>>>(
        flat.data_ptr<float>(), output.data_ptr<float>(), rows, cols, 1);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_var_kernel", KERNEL_SRC, ["aten_var_fwd"])
    x = torch.randn(32, 64, device="cuda")
    result = ext.aten_var_fwd(x)
    expected = aten.var.correction(x, [-1])
    check("aten.var", result, expected, atol=1e-3)
    print("PASS aten.var")

if __name__ == "__main__":
    test()
