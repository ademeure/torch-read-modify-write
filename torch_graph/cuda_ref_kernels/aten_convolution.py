"""Reference CUDA kernel for aten.convolution — naive conv2d with nested loops."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

// Naive conv2d: one thread per output element, no optimization
extern "C" __global__ void aten_conv2d_kernel(
    const float *input, const float *weight, const float *bias, float *output,
    unsigned int N, unsigned int C_in, unsigned int H, unsigned int W,
    unsigned int C_out, unsigned int kH, unsigned int kW,
    unsigned int padH, unsigned int padW, unsigned int strideH, unsigned int strideW,
    unsigned int outH, unsigned int outW
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = N * C_out * outH * outW;
    if (idx >= total) return;
    unsigned int ow = idx % outW;
    unsigned int oh = (idx / outW) % outH;
    unsigned int oc = (idx / (outW * outH)) % C_out;
    unsigned int n = idx / (outW * outH * C_out);

    float sum = (bias != nullptr) ? bias[oc] : 0.0f;
    for (unsigned int ic = 0; ic < C_in; ic++) {
        for (unsigned int kh = 0; kh < kH; kh++) {
            for (unsigned int kw = 0; kw < kW; kw++) {
                int ih = (int)(oh * strideH + kh) - (int)padH;
                int iw = (int)(ow * strideW + kw) - (int)padW;
                if (ih >= 0 && ih < (int)H && iw >= 0 && iw < (int)W) {
                    sum += input[n*C_in*H*W + ic*H*W + ih*W + iw]
                         * weight[oc*C_in*kH*kW + ic*kH*kW + kh*kW + kw];
                }
            }
        }
    }
    output[idx] = sum;
}

torch::Tensor aten_conv2d_fwd(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int64_t strideH, int64_t strideW, int64_t padH, int64_t padW
) {
    auto ci = input.contiguous();
    int N = ci.size(0), C_in = ci.size(1), H = ci.size(2), W = ci.size(3);
    int C_out = weight.size(0), kH = weight.size(2), kW = weight.size(3);
    int outH = (H + 2*padH - kH) / strideH + 1;
    int outW = (W + 2*padW - kW) / strideW + 1;
    auto output = torch::empty({N, C_out, outH, outW}, ci.options());
    int total = N * C_out * outH * outW;
    aten_conv2d_kernel<<<(total+255)/256, 256>>>(
        ci.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C_in, H, W, C_out, kH, kW, padH, padW, strideH, strideW, outH, outW);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_conv2d_kernel", KERNEL_SRC, ["aten_conv2d_fwd"])
    x = torch.randn(1, 3, 8, 8, device="cuda")
    w = torch.randn(16, 3, 3, 3, device="cuda")
    b = torch.randn(16, device="cuda")
    result = ext.aten_conv2d_fwd(x, w, b, 1, 1, 1, 1)
    expected = aten.convolution.default(x, w, b, [1,1], [1,1], [1,1], False, [0,0], 1)
    check("aten.convolution", result, expected, atol=1e-3)
    print("PASS aten.convolution")

if __name__ == "__main__":
    test()
