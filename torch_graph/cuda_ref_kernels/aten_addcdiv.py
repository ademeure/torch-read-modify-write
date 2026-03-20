"""Reference CUDA kernel for aten.addcdiv — input + value * tensor1 / tensor2."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_addcdiv(
    const float *inp, const float *t1, const float *t2,
    float *out, float value, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = inp[i] + value * t1[i] / t2[i];
    }
}

torch::Tensor aten_addcdiv_fwd(
    torch::Tensor inp, torch::Tensor t1, torch::Tensor t2, double value
) {
    auto out = torch::empty_like(inp);
    int n = inp.numel();
    aten_addcdiv<<<(n+255)/256, 256>>>(
        inp.data_ptr<float>(), t1.data_ptr<float>(), t2.data_ptr<float>(),
        out.data_ptr<float>(), (float)value, n);
    return out;
}
"""

def test():
    ext = compile_cuda("aten_addcdiv", KERNEL_SRC, ["aten_addcdiv_fwd"])
    inp = torch.randn(1024, device='cuda')
    t1 = torch.randn(1024, device='cuda')
    t2 = torch.randn(1024, device='cuda').abs() + 0.1
    result = ext.aten_addcdiv_fwd(inp, t1, t2, 0.5)
    expected = aten.addcdiv.default(inp, t1, t2, value=0.5)
    check("aten.addcdiv", result, expected, atol=1e-4)
    print("PASS aten.addcdiv")

if __name__ == "__main__":
    test()
