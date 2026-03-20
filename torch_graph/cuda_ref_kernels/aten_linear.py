"""Reference CUDA kernel for aten.linear — y = x @ weight.T + bias."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_linear(
    const float *x, const float *w, const float *bias, float *out,
    unsigned int M, unsigned int K, unsigned int N
) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = (bias != nullptr) ? bias[col] : 0.0f;
        for (unsigned int k = 0; k < K; k++)
            sum += x[row * K + k] * w[col * K + k];  // w is transposed
        out[row * N + col] = sum;
    }
}

torch::Tensor aten_linear_fwd(torch::Tensor x, torch::Tensor w, torch::Tensor bias) {
    int M = x.size(0), K = x.size(1), N = w.size(0);
    auto out = torch::empty({M, N}, x.options());
    dim3 threads(16, 16);
    dim3 blocks((N+15)/16, (M+15)/16);
    aten_linear<<<blocks, threads>>>(
        x.data_ptr<float>(), w.data_ptr<float>(), bias.data_ptr<float>(),
        out.data_ptr<float>(), M, K, N);
    return out;
}
"""

def test():
    ext = compile_cuda("aten_linear", KERNEL_SRC, ["aten_linear_fwd"])
    x = torch.randn(32, 64, device="cuda")
    w = torch.randn(48, 64, device="cuda")
    b = torch.randn(48, device="cuda")
    result = ext.aten_linear_fwd(x.contiguous(), w.contiguous(), b.contiguous())
    expected = aten.linear.default(x, w, b)
    check("aten.linear", result, expected, atol=1e-3)
    print("PASS aten.linear")

if __name__ == "__main__":
    test()
