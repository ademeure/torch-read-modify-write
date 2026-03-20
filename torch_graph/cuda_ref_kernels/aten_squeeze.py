"""Reference CUDA kernel for aten.squeeze — remove size-1 dimensions.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_squeeze.py --once
"""
import torch

KERNEL_SRC = r"""
extern "C" __global__ void aten_squeeze_copy(const float *in0, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out0[i] = in0[i];
}
"""

def init_once():
    x = torch.randn(32, 1, 64, device='cuda')
    return {
        "kernel_source": KERNEL_SRC,
        "inputs": [x.contiguous()],
        "expected": [torch.ops.aten.squeeze.dim(x, 1).contiguous().flatten()],
    }

def run(inputs, kernel):
    return [kernel(*inputs)]
