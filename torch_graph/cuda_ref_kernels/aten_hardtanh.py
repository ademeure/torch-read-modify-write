"""Reference CUDA kernel for aten.hardtanh.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_hardtanh.py --once
"""
import torch

KERNEL_SRC = r"""
extern "C" __global__ void aten_hardtanh(const float *in0, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in0[i];
        out0[i] = fminf(fmaxf(x, -1.0f), 1.0f);
    }
}
"""

def init_once():
    x = torch.randn(1024, device='cuda') * 3
    return {
        "kernel_source": KERNEL_SRC,
        "inputs": [x],
        "expected": [torch.ops.aten.hardtanh.default(x)],
        "atol": 1e-05,
    }

def run(inputs, kernel):
    return [kernel(*inputs)]
