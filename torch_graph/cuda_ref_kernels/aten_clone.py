"""Reference CUDA kernel for aten.clone — copy tensor to new contiguous memory."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_clone_kernel(const float *in, float *out, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i];
}

torch::Tensor aten_clone_fwd(torch::Tensor input) {
    auto ci = input.contiguous();
    auto output = torch::empty_like(ci);
    int n = ci.numel();
    aten_clone_kernel<<<(n+255)/256, 256>>>(
        ci.data_ptr<float>(), output.data_ptr<float>(), n);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_clone_kernel", KERNEL_SRC, ["aten_clone_fwd"])
    x = torch.randn(1024, device="cuda")
    result = ext.aten_clone_fwd(x)
    expected = aten.clone.default(x)
    check("aten.clone", result, expected)
    print("PASS aten.clone")

if __name__ == "__main__":
    test()
