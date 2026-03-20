"""Reference CUDA kernel for aten.unsqueeze — add dimension (contiguous copy)."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_unsqueeze_copy(const float *in, float *out, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i];
}

torch::Tensor aten_unsqueeze_fwd(torch::Tensor input, int64_t dim) {
    auto ci = input.contiguous();
    // Compute output shape
    auto sizes = ci.sizes().vec();
    if (dim < 0) dim = sizes.size() + 1 + dim;
    sizes.insert(sizes.begin() + dim, 1);
    auto output = torch::empty(sizes, ci.options());
    int n = ci.numel();
    aten_unsqueeze_copy<<<(n+255)/256, 256>>>(
        ci.data_ptr<float>(), output.data_ptr<float>(), n);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_unsqueeze_copy", KERNEL_SRC, ["aten_unsqueeze_fwd"])
    x = torch.randn(32, 64, device="cuda")
    result = ext.aten_unsqueeze_fwd(x, 0)
    expected = aten.unsqueeze.default(x, 0).contiguous()
    check("aten.unsqueeze", result, expected)
    print("PASS aten.unsqueeze")

if __name__ == "__main__":
    test()
