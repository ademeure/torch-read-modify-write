"""Reference CUDA kernel for aten.amax."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void aten_amax(
    const float *input, float *output, unsigned int rows, unsigned int cols
) {
    extern __shared__ float sdata[];
    unsigned int row = blockIdx.x, tid = threadIdx.x;
    if (row >= rows) return;
    const float *ri = input + row * cols;
    float v = -1e38f;
    for (unsigned int j = tid; j < cols; j += blockDim.x) {
        v = fmaxf(v, ri[j]);
    }
    sdata[tid] = v; __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) { sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]); } __syncthreads();
    }
    if (tid == 0) output[row] = sdata[0];
}

torch::Tensor aten_amax_fwd(torch::Tensor input, int dim) {
    auto sizes = input.sizes();
    int cols = sizes[sizes.size() - 1];
    int rows = input.numel() / cols;
    auto flat = input.reshape({rows, cols}).contiguous();
    auto output = torch::empty({rows}, input.options());
    int threads = 256;
    aten_amax<<<rows, threads, threads * sizeof(float)>>>(
        flat.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
    auto out_sizes = sizes.vec();
    out_sizes.pop_back();
    if (out_sizes.empty()) out_sizes.push_back(1);
    return output.reshape(out_sizes);
}
"""

def test():
    ext = compile_cuda("aten_amax", KERNEL_SRC, ["aten_amax_fwd"])
    x = torch.randn(32, 64, device='cuda')
    result = ext.aten_amax_fwd(x, -1)
    expected = aten.amax.default(x, [-1])
    check("aten.amax", result, expected)
    print(f"PASS aten.amax")

if __name__ == "__main__":
    test()
