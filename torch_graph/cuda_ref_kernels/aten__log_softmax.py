"""Reference CUDA kernel for aten._log_softmax — log(softmax(x))."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void aten_log_softmax(
    const float *input, float *output, unsigned int rows, unsigned int cols
) {
    extern __shared__ float sdata[];
    unsigned int row = blockIdx.x, tid = threadIdx.x;
    if (row >= rows) return;
    const float *ri = input + row * cols;
    float *ro = output + row * cols;
    float v = -1e38f;
    for (unsigned int j = tid; j < cols; j += blockDim.x) v = fmaxf(v, ri[j]);
    sdata[tid] = v; __syncthreads();
    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid+s]); __syncthreads();
    }
    float row_max = sdata[0];
    float lsum = 0.0f;
    for (unsigned int j = tid; j < cols; j += blockDim.x)
        lsum += expf(ri[j] - row_max);
    sdata[tid] = lsum; __syncthreads();
    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid+s]; __syncthreads();
    }
    float log_sum = logf(sdata[0]);
    for (unsigned int j = tid; j < cols; j += blockDim.x)
        ro[j] = (ri[j] - row_max) - log_sum;
}

torch::Tensor aten_log_softmax_fwd(torch::Tensor input) {
    auto sizes = input.sizes();
    int cols = sizes[sizes.size() - 1];
    int rows = input.numel() / cols;
    auto flat = input.reshape({rows, cols}).contiguous();
    auto output = torch::empty_like(flat);
    int threads = 256;
    aten_log_softmax<<<rows, threads, threads * sizeof(float)>>>(
        flat.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
    return output.reshape(sizes);
}
"""

def test():
    ext = compile_cuda("aten_log_softmax", KERNEL_SRC, ["aten_log_softmax_fwd"])
    x = torch.randn(8, 64, device="cuda")
    result = ext.aten_log_softmax_fwd(x)
    expected = aten._log_softmax.default(x, -1, False)
    check("aten._log_softmax", result, expected, atol=1e-5)
    print("PASS aten._log_softmax")

if __name__ == "__main__":
    test()
