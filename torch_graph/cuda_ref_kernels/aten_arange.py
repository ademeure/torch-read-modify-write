"""Reference CUDA kernel for aten.arange — fill with sequential values."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_arange_kernel(
    float *output, float start, float step, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = start + i * step;
}

torch::Tensor aten_arange_fwd(double start, double end, double step) {
    int n = (int)ceil((end - start) / step);
    auto output = torch::empty({n}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    aten_arange_kernel<<<(n+255)/256, 256>>>(
        output.data_ptr<float>(), (float)start, (float)step, n);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_arange_kernel", KERNEL_SRC, ["aten_arange_fwd"])
    result = ext.aten_arange_fwd(0.0, 100.0, 1.0)
    expected = aten.arange.start_step(0, 100, 1, dtype=torch.float32, device='cuda')
    check("aten.arange", result, expected)
    print("PASS aten.arange")

if __name__ == "__main__":
    test()
