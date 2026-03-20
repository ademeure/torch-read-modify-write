"""Reference CUDA kernel for aten.linspace — evenly spaced values."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_linspace_kernel(
    float *output, float start, float step, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = start + i * step;
}

torch::Tensor aten_linspace_fwd(double start, double end, int64_t steps) {
    auto output = torch::empty({steps}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    float step = (steps > 1) ? (float)(end - start) / (steps - 1) : 0.0f;
    aten_linspace_kernel<<<(steps+255)/256, 256>>>(
        output.data_ptr<float>(), (float)start, step, steps);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_linspace_kernel", KERNEL_SRC, ["aten_linspace_fwd"])
    result = ext.aten_linspace_fwd(0.0, 1.0, 100)
    expected = torch.linspace(0, 1, 100, device='cuda')
    check("aten.linspace", result, expected, atol=1e-5)
    print("PASS aten.linspace")

if __name__ == "__main__":
    test()
