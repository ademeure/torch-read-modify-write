"""Reference CUDA kernel for aten.dot — inner product of two vectors."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_dot(
    const float *a, const float *b, float *out, unsigned int n
) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    float v = 0.0f;
    for (unsigned int i = tid; i < n; i += blockDim.x)
        v += a[i] * b[i];
    sdata[tid] = v;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[0] = sdata[0];
}

torch::Tensor aten_dot_fwd(torch::Tensor a, torch::Tensor b) {
    auto out = torch::zeros({}, a.options());
    int n = a.numel();
    int threads = 256;
    aten_dot<<<1, threads, threads * sizeof(float)>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), n);
    return out;
}
"""

def test():
    ext = compile_cuda("aten_dot", KERNEL_SRC, ["aten_dot_fwd"])
    a = torch.randn(256, device="cuda")
    b = torch.randn(256, device="cuda")
    result = ext.aten_dot_fwd(a, b)
    expected = aten.dot.default(a, b)
    check("aten.dot", result, expected, atol=1e-3)
    print("PASS aten.dot")

if __name__ == "__main__":
    test()
