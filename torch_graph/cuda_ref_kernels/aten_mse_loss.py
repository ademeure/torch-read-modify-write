"""Reference CUDA kernel for aten.mse_loss — mean squared error."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_mse_kernel(
    const float *input, const float *target, float *output, unsigned int n
) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    float v = 0.0f;
    for (unsigned int i = tid; i < n; i += blockDim.x) {
        float d = input[i] - target[i];
        v += d * d;
    }
    sdata[tid] = v; __syncthreads();
    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid+s]; __syncthreads();
    }
    if (tid == 0) output[0] = sdata[0] / (float)n;
}

torch::Tensor aten_mse_fwd(torch::Tensor input, torch::Tensor target) {
    auto ci = input.contiguous();
    auto ct = target.contiguous();
    int n = ci.numel();
    auto output = torch::zeros({}, ci.options());
    int threads = 256;
    aten_mse_kernel<<<1, threads, threads * sizeof(float)>>>(
        ci.data_ptr<float>(), ct.data_ptr<float>(), output.data_ptr<float>(), n);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_mse_kernel", KERNEL_SRC, ["aten_mse_fwd"])
    x = torch.randn(256, device="cuda")
    y = torch.randn(256, device="cuda")
    result = ext.aten_mse_fwd(x, y)
    expected = aten.mse_loss.default(x, y)
    check("aten.mse_loss", result, expected, atol=1e-4)
    print("PASS aten.mse_loss")

if __name__ == "__main__":
    test()
