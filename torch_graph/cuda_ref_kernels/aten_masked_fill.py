"""Reference CUDA kernel for aten.masked_fill — fill where mask is True."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_masked_fill(
    const float *input, const float *mask, float *out, float value, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = (mask[i] != 0.0f) ? value : input[i];
    }
}

torch::Tensor aten_masked_fill_fwd(torch::Tensor input, torch::Tensor mask, double value) {
    auto out = torch::empty_like(input);
    int n = input.numel();
    aten_masked_fill<<<(n+255)/256, 256>>>(
        input.data_ptr<float>(), mask.data_ptr<float>(), out.data_ptr<float>(),
        (float)value, n);
    return out;
}
"""

def test():
    ext = compile_cuda("aten_masked_fill", KERNEL_SRC, ["aten_masked_fill_fwd"])
    x = torch.randn(1024, device='cuda')
    mask = (torch.randn(1024, device='cuda') > 0).float()
    result = ext.aten_masked_fill_fwd(x, mask, -1e9)
    expected = aten.masked_fill.Scalar(x, mask.bool(), -1e9)
    check("aten.masked_fill", result, expected)
    print("PASS aten.masked_fill")

if __name__ == "__main__":
    test()
