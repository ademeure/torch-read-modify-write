"""Reference CUDA kernel for aten.clamp — clamp to [min, max] range.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_clamp.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_clamp(
    const float *in0, float *out0, float lo, float hi, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out0[i] = fminf(fmaxf(in0[i], lo), hi);
}
"""

def init_once():
    x = torch.randn(1024, device='cuda') * 5
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x],
        "expected": [torch.ops.aten.clamp.default(x, -1.0, 1.0)],
    }

def run(inputs, kernel):
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.float32(-1.0), np.float32(1.0), np.uint32(inputs[0].numel()),
    ])]
