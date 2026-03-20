"""Reference CUDA kernel for aten.constant_pad_nd — pad with constant value."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

// 2D constant padding: pad last 2 dims
extern "C" __global__ void aten_constant_pad_2d(
    const float *input, float *output,
    unsigned int H, unsigned int W, unsigned int outH, unsigned int outW,
    unsigned int padTop, unsigned int padLeft, float value, unsigned int batch_stride
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = batch_stride;  // N * outH * outW
    if (idx >= total) return;
    unsigned int ow = idx % outW;
    unsigned int oh = (idx / outW) % outH;
    unsigned int n = idx / (outH * outW);

    int ih = (int)oh - (int)padTop;
    int iw = (int)ow - (int)padLeft;
    if (ih >= 0 && ih < (int)H && iw >= 0 && iw < (int)W)
        output[idx] = input[n * H * W + ih * W + iw];
    else
        output[idx] = value;
}

torch::Tensor aten_constant_pad_2d_fwd(
    torch::Tensor input, int64_t padL, int64_t padR, int64_t padT, int64_t padB, double value
) {
    auto ci = input.contiguous();
    int N = ci.numel() / (ci.size(-2) * ci.size(-1));
    int H = ci.size(-2), W = ci.size(-1);
    int outH = H + padT + padB, outW = W + padL + padR;
    auto sizes = ci.sizes().vec();
    sizes[sizes.size()-2] = outH;
    sizes[sizes.size()-1] = outW;
    auto output = torch::empty(sizes, ci.options());
    int total = N * outH * outW;
    aten_constant_pad_2d<<<(total+255)/256, 256>>>(
        ci.data_ptr<float>(), output.data_ptr<float>(),
        H, W, outH, outW, padT, padL, (float)value, total);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_constant_pad_2d", KERNEL_SRC, ["aten_constant_pad_2d_fwd"])
    x = torch.randn(2, 8, 8, device="cuda")
    result = ext.aten_constant_pad_2d_fwd(x, 1, 1, 1, 1, 0.0)
    expected = aten.constant_pad_nd.default(x, [1, 1, 1, 1], 0.0)
    check("aten.constant_pad_nd", result, expected)
    print("PASS aten.constant_pad_nd")

if __name__ == "__main__":
    test()
