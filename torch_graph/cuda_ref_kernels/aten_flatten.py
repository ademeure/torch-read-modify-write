"""Reference CUDA kernel for aten.flatten — flatten dims [start, end] to single dim."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_flatten_copy(const float *in, float *out, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i];
}

torch::Tensor aten_flatten_fwd(torch::Tensor input, int64_t start, int64_t end) {
    auto ci = input.contiguous();
    auto t = ci.flatten(start, end);
    auto output = torch::empty(t.sizes().vec(), ci.options());
    int n = ci.numel();
    aten_flatten_copy<<<(n+255)/256, 256>>>(
        ci.data_ptr<float>(), output.data_ptr<float>(), n);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_flatten_copy", KERNEL_SRC, ["aten_flatten_fwd"])
    x = torch.randn(4, 8, 16, device="cuda")
    result = ext.aten_flatten_fwd(x, 1, 2)
    expected = aten.flatten.using_ints(x, 1, 2).contiguous()
    check("aten.flatten", result, expected)
    print("PASS aten.flatten")

if __name__ == "__main__":
    test()
