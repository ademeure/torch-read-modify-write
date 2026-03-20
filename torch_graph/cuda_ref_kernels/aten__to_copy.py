"""Reference CUDA kernel for aten._to_copy — dtype/device conversion (copy).
Run: kbox iterate torch_graph/cuda_ref_kernels/aten__to_copy.py --once
"""
import torch

KERNEL_SRC = r"""
extern "C" __global__ void aten__to_copy_copy(const float *in0, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out0[i] = in0[i];
}
"""

def init_once():
    x = torch.randn(1024, device='cuda')
    return {
        "kernel_source": KERNEL_SRC,
        "inputs": [x.contiguous()],
        "expected": [torch.ops.aten._to_copy.default(x).flatten()],
    }

def run(inputs, kernel):
    return [kernel(*inputs)]
