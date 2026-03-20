"""Reference CUDA kernel for aten.native_dropout — zero mask with scaling."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

// Deterministic "dropout" for testing: use mask from Python side
extern "C" __global__ void aten_dropout_kernel(
    const float *input, const float *mask, float *output,
    float scale, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = input[i] * mask[i] * scale;
}

torch::Tensor aten_dropout_fwd(torch::Tensor input, torch::Tensor mask, double p) {
    auto ci = input.contiguous();
    int n = ci.numel();
    float scale = 1.0f / (1.0f - (float)p);
    auto output = torch::empty_like(ci);
    aten_dropout_kernel<<<(n+255)/256, 256>>>(
        ci.data_ptr<float>(), mask.data_ptr<float>(), output.data_ptr<float>(), scale, n);
    return output;
}
"""

def test():
    # Can't compare directly against native_dropout (random), so verify the math
    ext = compile_cuda("aten_dropout_kernel", KERNEL_SRC, ["aten_dropout_fwd"])
    x = torch.randn(1024, device="cuda")
    mask = (torch.rand(1024, device="cuda") > 0.5).float()
    result = ext.aten_dropout_fwd(x, mask, 0.5)
    expected = x * mask * 2.0  # scale = 1/(1-0.5) = 2
    check("aten.native_dropout", result, expected)
    print("PASS aten.native_dropout")

if __name__ == "__main__":
    test()
