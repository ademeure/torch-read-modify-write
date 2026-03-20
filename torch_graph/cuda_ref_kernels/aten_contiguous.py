"""Reference CUDA kernel for aten.contiguous — ensure contiguous memory layout.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_contiguous.py --once
"""
import torch

KERNEL_SRC = r"""
extern "C" __global__ void aten_contiguous_copy(const float *in0, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out0[i] = in0[i];
}
"""

def init_once():
    x = torch.randn(32, 64, device='cuda').t().contiguous()
    return {
        "kernel_source": KERNEL_SRC,
        "inputs": [x.contiguous()],
        "expected": [x.clone().flatten()],
    }

def run(inputs, kernel):
    return [kernel(*inputs)]
