"""Reference CUDA kernel for aten.gelu_backward (backward gradient op)."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void aten_gelu_backward(const float *grad, const float *saved, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = grad[i];
        float s = saved[i];
        out0[i] = (g * (0.5f * (1.0f + erff(s * 0.7071067811865476f)) + s * 0.3989422804014327f * expf(-0.5f * s * s)));
    }
}

torch::Tensor aten_gelu_backward_fwd(torch::Tensor grad, torch::Tensor saved) {
    auto out0 = torch::empty_like(grad);
    int n = grad.numel();
    aten_gelu_backward<<<(n+255)/256, 256>>>(grad.data_ptr<float>(), saved.data_ptr<float>(), out0.data_ptr<float>(), n);
    return out0;
}
"""

def test():
    ext = compile_cuda("aten_gelu_backward", KERNEL_SRC, ["aten_gelu_backward_fwd"])
    grad = torch.randn(1024, device='cuda')
    saved = torch.randn(1024, device='cuda')
    result = ext.aten_gelu_backward_fwd(grad, saved)
    expected = aten.gelu_backward.default(grad, saved)
    check("aten.gelu_backward", result, expected, atol=0.0001)
    print(f"PASS aten.gelu_backward")

if __name__ == "__main__":
    test()
