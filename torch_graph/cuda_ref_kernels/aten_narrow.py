"""Reference CUDA kernel for aten.narrow — narrow view along a dimension.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_narrow.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_narrow_copy(
    const float *input, float *output, unsigned int cols,
    unsigned int start, unsigned int length
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= length * cols) return;
    unsigned int r = idx / cols, c = idx % cols;
    output[idx] = input[(start + r) * cols + c];
}
"""

def init_once():
    x = torch.randn(32, 64, device="cuda")
    total = 10 * 64
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x],
        "expected": [torch.ops.aten.narrow.default(x, 0, 4, 10).contiguous().flatten()],
        "outputs": ["float32;n=%d" % total],
        "grid": ((total + 255) // 256,),
    }

def run(inputs, kernel):
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(64), np.uint32(4), np.uint32(10),
    ])]
