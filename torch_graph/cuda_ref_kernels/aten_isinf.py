"""Reference CUDA kernel for aten.isinf.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_isinf.py --once
"""
import torch

KERNEL_SRC = r"""
extern "C" __global__ void aten_isinf(const float *in0, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in0[i];
        out0[i] = (isinf(x) ? 1.0f : 0.0f);
    }
}
"""

def init_once():
    x = torch.tensor([1.0, float('inf'), 0.0, float('-inf'), -1.0] * 200, device='cuda')
    return {
        "kernel_source": KERNEL_SRC,
        "inputs": [x],
        "expected": [torch.ops.aten.isinf.default(x).float()],
    }

def run(inputs, kernel):
    return [kernel(*inputs)]
