"""Reference CUDA kernel for aten.avg_pool2d — average pooling."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_avg_pool2d_kernel(
    const float *input, float *output,
    unsigned int N, unsigned int C, unsigned int H, unsigned int W,
    unsigned int kH, unsigned int kW, unsigned int strideH, unsigned int strideW,
    unsigned int padH, unsigned int padW, unsigned int outH, unsigned int outW
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = N * C * outH * outW;
    if (idx >= total) return;
    unsigned int ow = idx % outW;
    unsigned int oh = (idx / outW) % outH;
    unsigned int c = (idx / (outW * outH)) % C;
    unsigned int n = idx / (outW * outH * C);

    float sum = 0.0f;
    int count = 0;
    for (unsigned int kh = 0; kh < kH; kh++) {
        for (unsigned int kw = 0; kw < kW; kw++) {
            int ih = (int)(oh * strideH + kh) - (int)padH;
            int iw = (int)(ow * strideW + kw) - (int)padW;
            if (ih >= 0 && ih < (int)H && iw >= 0 && iw < (int)W) {
                sum += input[n*C*H*W + c*H*W + ih*W + iw];
                count++;
            }
        }
    }
    output[idx] = sum / (float)count;
}

torch::Tensor aten_avg_pool2d_fwd(
    torch::Tensor input, int64_t kH, int64_t kW,
    int64_t strideH, int64_t strideW, int64_t padH, int64_t padW
) {
    auto ci = input.contiguous();
    int N = ci.size(0), C = ci.size(1), H = ci.size(2), W = ci.size(3);
    int outH = (H + 2*padH - kH) / strideH + 1;
    int outW = (W + 2*padW - kW) / strideW + 1;
    auto output = torch::empty({N, C, outH, outW}, ci.options());
    int total = N * C * outH * outW;
    aten_avg_pool2d_kernel<<<(total+255)/256, 256>>>(
        ci.data_ptr<float>(), output.data_ptr<float>(),
        N, C, H, W, kH, kW, strideH, strideW, padH, padW, outH, outW);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_avg_pool2d_kernel", KERNEL_SRC, ["aten_avg_pool2d_fwd"])
    x = torch.randn(1, 4, 8, 8, device="cuda")
    result = ext.aten_avg_pool2d_fwd(x, 2, 2, 2, 2, 0, 0)
    expected = aten.avg_pool2d.default(x, [2,2], [2,2])
    check("aten.avg_pool2d", result, expected, atol=1e-5)
    print("PASS aten.avg_pool2d")

if __name__ == "__main__":
    test()
