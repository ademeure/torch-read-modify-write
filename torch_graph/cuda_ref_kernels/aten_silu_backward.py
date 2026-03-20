"""Reference CUDA kernel for aten.silu_backward (backward gradient op)."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void aten_silu_backward(const float *grad, const float *saved, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = grad[i];
        float s = saved[i];
        out0[i] = (g * (1.0f / (1.0f + expf(-s))) * (1.0f + s * (1.0f - 1.0f / (1.0f + expf(-s)))));
    }
}

torch::Tensor aten_silu_backward_fwd(torch::Tensor grad, torch::Tensor saved) {
    auto out0 = torch::empty_like(grad);
    int n = grad.numel();
    aten_silu_backward<<<(n+255)/256, 256>>>(grad.data_ptr<float>(), saved.data_ptr<float>(), out0.data_ptr<float>(), n);
    return out0;
}
"""

def test():
    ext = compile_cuda("aten_silu_backward", KERNEL_SRC, ["aten_silu_backward_fwd"])
    grad = torch.randn(1024, device='cuda')
    saved = torch.randn(1024, device='cuda')
    result = ext.aten_silu_backward_fwd(grad, saved)
    expected = aten.silu_backward.default(grad, saved)
    check("aten.silu_backward", result, expected, atol=0.0001)
    print(f"PASS aten.silu_backward")

if __name__ == "__main__":
    test()
