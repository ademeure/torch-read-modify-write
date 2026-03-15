"""Reference CUDA kernel for aten.relu."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

KERNEL_SRC = r"""
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void aten_relu(const float *in0, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in0[i];
        out0[i] = ((x > 0.0f) ? x : 0.0f);
    }
}

torch::Tensor aten_relu_fwd(torch::Tensor in0) {
    auto out0 = torch::empty_like(in0);
    int n = in0.numel();
    aten_relu<<<(n+255)/256, 256>>>(in0.data_ptr<float>(), out0.data_ptr<float>(), n);
    return out0;
}
"""

def test():
    ext = compile_cuda("aten_relu", KERNEL_SRC, ["aten_relu_fwd"])
    x = torch.randn(1024, device='cuda')
    result = ext.aten_relu_fwd(x)
    expected = x.relu()
    check("aten.relu", result, expected)
    print(f"PASS aten.relu")

if __name__ == "__main__":
    test()
