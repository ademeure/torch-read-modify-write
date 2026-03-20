"""Reference CUDA kernel for aten.div."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void aten_div(const float *in0, const float *in1, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float a = in0[i];
        float b = in1[i];
        out0[i] = (a / b);
    }
}

torch::Tensor aten_div_fwd(torch::Tensor in0, torch::Tensor in1) {
    auto out0 = torch::empty_like(in0);
    int n = in0.numel();
    aten_div<<<(n+255)/256, 256>>>(in0.data_ptr<float>(), in1.data_ptr<float>(), out0.data_ptr<float>(), n);
    return out0;
}
"""

def test():
    ext = compile_cuda("aten_div", KERNEL_SRC, ["aten_div_fwd"])
    a = torch.randn(1024, device='cuda')
    b = torch.randn(1024, device='cuda').abs() + 0.1
    result = ext.aten_div_fwd(a, b)
    expected = aten.div.Tensor(a, b)
    check("aten.div", result, expected, atol=1e-05)
    print(f"PASS aten.div")

if __name__ == "__main__":
    test()
