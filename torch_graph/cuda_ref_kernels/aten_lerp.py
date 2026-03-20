"""Reference CUDA kernel for aten.lerp — linear interpolation: start + weight*(end-start)."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_lerp(
    const float *a, const float *b, const float *w, float *out, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + w[i] * (b[i] - a[i]);
    }
}

torch::Tensor aten_lerp_fwd(torch::Tensor a, torch::Tensor b, torch::Tensor w) {
    auto out = torch::empty_like(a);
    int n = a.numel();
    aten_lerp<<<(n+255)/256, 256>>>(
        a.data_ptr<float>(), b.data_ptr<float>(),
        w.data_ptr<float>(), out.data_ptr<float>(), n);
    return out;
}
"""

def test():
    ext = compile_cuda("aten_lerp", KERNEL_SRC, ["aten_lerp_fwd"])
    a = torch.randn(1024, device='cuda')
    b = torch.randn(1024, device='cuda')
    w = torch.rand(1024, device='cuda')
    result = ext.aten_lerp_fwd(a, b, w)
    expected = aten.lerp.Tensor(a, b, w)
    check("aten.lerp", result, expected, atol=1e-5)
    print("PASS aten.lerp")

if __name__ == "__main__":
    test()
