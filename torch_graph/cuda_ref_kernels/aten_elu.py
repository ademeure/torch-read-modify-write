"""Reference CUDA kernel for aten.elu."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

KERNEL_SRC = r"""
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void aten_elu(const float *in0, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in0[i];
        out0[i] = ((x > 0.0f) ? x : (expf(x) - 1.0f));
    }
}

torch::Tensor aten_elu_fwd(torch::Tensor in0) {
    auto out0 = torch::empty_like(in0);
    int n = in0.numel();
    aten_elu<<<(n+255)/256, 256>>>(in0.data_ptr<float>(), out0.data_ptr<float>(), n);
    return out0;
}
"""

def test():
    ext = compile_cuda("aten_elu", KERNEL_SRC, ["aten_elu_fwd"])
    x = torch.randn(1024, device='cuda')
    result = ext.aten_elu_fwd(x)
    expected = torch.nn.functional.elu(x)
    check("aten.elu", result, expected, atol=1e-05)
    print(f"PASS aten.elu")

if __name__ == "__main__":
    test()
