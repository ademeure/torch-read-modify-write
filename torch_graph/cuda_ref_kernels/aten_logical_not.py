"""Reference CUDA kernel for aten.logical_not.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_logical_not.py --once
"""
import torch

KERNEL_SRC = r"""
extern "C" __global__ void aten_logical_not(const float *in0, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in0[i];
        out0[i] = ((x == 0.0f) ? 1.0f : 0.0f);
    }
}
"""

def init_once():
    x = torch.tensor([1.0, 0.0, -1.0, 0.0, 3.14] * 200, device='cuda')
    return {
        "kernel_source": KERNEL_SRC,
        "inputs": [x],
        "expected": [torch.ops.aten.logical_not.default(x).float()],
    }

def run(inputs, kernel):
    return [kernel(*inputs)]
