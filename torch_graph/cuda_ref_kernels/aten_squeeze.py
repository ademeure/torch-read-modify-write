"""Reference CUDA kernel for aten.squeeze — remove size-1 dimensions (contiguous copy)."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_squeeze_copy(const float *in, float *out, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i];
}

torch::Tensor aten_squeeze_fwd(torch::Tensor input, int64_t dim) {
    auto ci = input.contiguous();
    auto sizes = ci.sizes().vec();
    if (dim < 0) dim = sizes.size() + dim;
    if (sizes[dim] == 1) sizes.erase(sizes.begin() + dim);
    auto output = torch::empty(sizes, ci.options());
    int n = ci.numel();
    aten_squeeze_copy<<<(n+255)/256, 256>>>(
        ci.data_ptr<float>(), output.data_ptr<float>(), n);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_squeeze_copy", KERNEL_SRC, ["aten_squeeze_fwd"])
    x = torch.randn(32, 1, 64, device="cuda")
    result = ext.aten_squeeze_fwd(x, 1)
    expected = aten.squeeze.dim(x, 1).contiguous()
    check("aten.squeeze", result, expected)
    print("PASS aten.squeeze")

if __name__ == "__main__":
    test()
