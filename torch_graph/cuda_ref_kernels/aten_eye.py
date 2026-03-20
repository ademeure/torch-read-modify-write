"""Reference CUDA kernel for aten.eye — identity matrix."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_eye_kernel(float *output, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * n) return;
    unsigned int r = idx / n, c = idx % n;
    output[idx] = (r == c) ? 1.0f : 0.0f;
}

torch::Tensor aten_eye_fwd(int64_t n) {
    auto output = torch::empty({n, n}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    int total = n * n;
    aten_eye_kernel<<<(total+255)/256, 256>>>(output.data_ptr<float>(), n);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_eye_kernel", KERNEL_SRC, ["aten_eye_fwd"])
    result = ext.aten_eye_fwd(32)
    expected = torch.eye(32, device='cuda')
    check("aten.eye", result, expected)
    print("PASS aten.eye")

if __name__ == "__main__":
    test()
