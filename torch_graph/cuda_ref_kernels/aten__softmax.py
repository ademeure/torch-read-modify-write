"""Reference CUDA kernel for aten._softmax."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

KERNEL_SRC = r"""
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void aten_softmax(
    const float *input, float *output, unsigned int rows, unsigned int cols
) {
    extern __shared__ float sdata[];
    unsigned int row = blockIdx.x, tid = threadIdx.x;
    if (row >= rows) return;
    const float *ri = input + row * cols;
    float *ro = output + row * cols;
    // Pass 1: max
    float v = -1e38f;
    for (unsigned int j = tid; j < cols; j += blockDim.x) v = fmaxf(v, ri[j]);
    sdata[tid] = v; __syncthreads();
    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid+s]); __syncthreads();
    }
    float row_max = sdata[0];
    // Pass 2: exp + sum
    float lsum = 0.0f;
    for (unsigned int j = tid; j < cols; j += blockDim.x) {
        float e = expf(ri[j] - row_max); ro[j] = e; lsum += e;
    }
    sdata[tid] = lsum; __syncthreads();
    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid+s]; __syncthreads();
    }
    float inv = 1.0f / sdata[0];
    // Pass 3: normalize
    for (unsigned int j = tid; j < cols; j += blockDim.x) ro[j] *= inv;
}

torch::Tensor aten_softmax_fwd(torch::Tensor input) {
    auto sizes = input.sizes();
    int cols = sizes[sizes.size() - 1];
    int rows = input.numel() / cols;
    auto flat = input.reshape({rows, cols}).contiguous();
    auto output = torch::empty_like(flat);
    int threads = 256;
    aten_softmax<<<rows, threads, threads * sizeof(float)>>>(
        flat.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
    return output.reshape(sizes);
}
"""

def test():
    ext = compile_cuda("aten_softmax", KERNEL_SRC, ["aten_softmax_fwd"])
    x = torch.randn(8, 64, device="cuda")
    result = ext.aten_softmax_fwd(x)
    expected = torch.softmax(x, dim=-1)
    check("aten._softmax", result, expected, atol=1e-5)
    print("PASS aten._softmax")

if __name__ == "__main__":
    test()
