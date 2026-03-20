"""Reference CUDA kernel for aten.log_sigmoid."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void aten_log_sigmoid_forward(const float *in0, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in0[i];
        out0[i] = (-logf(1.0f + expf(-x)));
    }
}

torch::Tensor aten_log_sigmoid_forward_fwd(torch::Tensor in0) {
    auto out0 = torch::empty_like(in0);
    int n = in0.numel();
    aten_log_sigmoid_forward<<<(n+255)/256, 256>>>(in0.data_ptr<float>(), out0.data_ptr<float>(), n);
    return out0;
}
"""

def test():
    ext = compile_cuda("aten_log_sigmoid_forward", KERNEL_SRC, ["aten_log_sigmoid_forward_fwd"])
    x = torch.randn(1024, device='cuda')
    result = ext.aten_log_sigmoid_forward_fwd(x)
    expected = aten.log_sigmoid_forward.default(x)[0]
    check("aten.log_sigmoid", result, expected, atol=1e-05)
    print(f"PASS aten.log_sigmoid")

if __name__ == "__main__":
    test()
