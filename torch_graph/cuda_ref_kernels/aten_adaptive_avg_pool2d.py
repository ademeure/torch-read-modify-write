"""Reference CUDA kernel for aten.adaptive_avg_pool2d."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_adaptive_avg_pool2d_kernel(
    const float *input, float *output,
    unsigned int N, unsigned int C, unsigned int H, unsigned int W,
    unsigned int outH, unsigned int outW
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = N * C * outH * outW;
    if (idx >= total) return;
    unsigned int ow = idx % outW;
    unsigned int oh = (idx / outW) % outH;
    unsigned int c = (idx / (outW * outH)) % C;
    unsigned int n = idx / (outW * outH * C);

    unsigned int h_start = oh * H / outH;
    unsigned int h_end = (oh + 1) * H / outH;
    unsigned int w_start = ow * W / outW;
    unsigned int w_end = (ow + 1) * W / outW;
    float sum = 0.0f;
    int count = 0;
    for (unsigned int h = h_start; h < h_end; h++) {
        for (unsigned int w = w_start; w < w_end; w++) {
            sum += input[n*C*H*W + c*H*W + h*W + w];
            count++;
        }
    }
    output[idx] = sum / (float)count;
}

torch::Tensor aten_adaptive_avg_pool2d_fwd(torch::Tensor input, int64_t outH, int64_t outW) {
    auto ci = input.contiguous();
    int N = ci.size(0), C = ci.size(1), H = ci.size(2), W = ci.size(3);
    auto output = torch::empty({N, C, (int)outH, (int)outW}, ci.options());
    int total = N * C * outH * outW;
    aten_adaptive_avg_pool2d_kernel<<<(total+255)/256, 256>>>(
        ci.data_ptr<float>(), output.data_ptr<float>(), N, C, H, W, outH, outW);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_adaptive_avg_pool2d_kernel", KERNEL_SRC, ["aten_adaptive_avg_pool2d_fwd"])
    x = torch.randn(1, 4, 8, 8, device="cuda")
    result = ext.aten_adaptive_avg_pool2d_fwd(x, 1, 1)
    expected = aten.adaptive_avg_pool2d.default(x, [1, 1])
    check("aten.adaptive_avg_pool2d", result, expected, atol=1e-4)
    print("PASS aten.adaptive_avg_pool2d")

if __name__ == "__main__":
    test()
