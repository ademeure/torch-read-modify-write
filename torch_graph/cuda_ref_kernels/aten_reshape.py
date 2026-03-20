"""Reference CUDA kernel for aten.reshape — reshape with contiguous output.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_reshape.py --once
"""
import torch

KERNEL_SRC = r"""
extern "C" __global__ void aten_reshape_copy(const float *in0, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out0[i] = in0[i];
}
"""

def init_once():
    x = torch.randn(32, 64, device='cuda')
    return {
        "kernel_source": KERNEL_SRC,
        "inputs": [x.contiguous()],
        "expected": [torch.ops.aten.reshape.default(x, [64, 32]).contiguous().flatten()],
    }

def run(inputs, kernel):
    return [kernel(*inputs)]
