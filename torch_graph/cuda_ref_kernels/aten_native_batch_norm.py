"""Reference CUDA kernel for aten.native_batch_norm — per-channel normalization."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void aten_batch_norm(
    const float *input, const float *weight, const float *bias,
    const float *running_mean, const float *running_var,
    float *output, unsigned int N, unsigned int C, unsigned int HW, float eps
) {
    // One thread per element. input shape: [N, C, HW]
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = N * C * HW;
    if (idx < total) {
        unsigned int c = (idx / HW) % C;
        float mean = running_mean[c];
        float var = running_var[c];
        float x = input[idx];
        float normed = (x - mean) * rsqrtf(var + eps);
        output[idx] = normed * weight[c] + bias[c];
    }
}

torch::Tensor aten_batch_norm_fwd(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor running_mean, torch::Tensor running_var, double eps
) {
    auto shape = input.sizes();
    int N = shape[0], C = shape[1];
    int HW = input.numel() / (N * C);
    auto flat = input.contiguous();
    auto output = torch::empty_like(flat);
    int total = flat.numel();
    aten_batch_norm<<<(total+255)/256, 256>>>(
        flat.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        running_mean.data_ptr<float>(), running_var.data_ptr<float>(),
        output.data_ptr<float>(), N, C, HW, (float)eps);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_batch_norm", KERNEL_SRC, ["aten_batch_norm_fwd"])
    x = torch.randn(2, 8, 4, 4, device="cuda")
    w = torch.randn(8, device="cuda")
    b = torch.randn(8, device="cuda")
    rm = torch.randn(8, device="cuda")
    rv = torch.rand(8, device="cuda") + 0.1
    result = ext.aten_batch_norm_fwd(x, w, b, rm, rv, 1e-5)
    expected = aten.native_batch_norm.default(x, w, b, rm, rv, False, 0.1, 1e-5)
    check("aten.native_batch_norm", result, expected[0], atol=1e-4)
    print("PASS aten.native_batch_norm")

if __name__ == "__main__":
    test()
