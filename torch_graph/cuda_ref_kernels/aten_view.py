"""Reference CUDA kernel for aten.view — reshape (data copy if non-contiguous)."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_view_copy(const float *in, float *out, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i];
}

torch::Tensor aten_view_fwd(torch::Tensor input, int64_t d0, int64_t d1) {
    auto ci = input.contiguous();
    auto output = torch::empty({d0, d1}, ci.options());
    int n = ci.numel();
    aten_view_copy<<<(n+255)/256, 256>>>(
        ci.data_ptr<float>(), output.data_ptr<float>(), n);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_view_copy", KERNEL_SRC, ["aten_view_fwd"])
    x = torch.randn(32, 64, device="cuda")
    result = ext.aten_view_fwd(x, 64, 32)
    expected = aten.view.default(x, [64, 32]).contiguous()
    check("aten.view", result, expected)
    print("PASS aten.view")

if __name__ == "__main__":
    test()
