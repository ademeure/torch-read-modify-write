"""Reference CUDA kernel for aten.transpose — 2D transpose, contiguous output."""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_transpose_2d(
    const float *in0, float *out0, unsigned int rows, unsigned int cols
) {
    unsigned int r = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows && c < cols) out0[c * rows + r] = in0[r * cols + c];
}
"""

ROWS, COLS = 32, 64

def init_once():
    x = torch.randn(ROWS, COLS, device="cuda")
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x],
        "expected": [torch.ops.aten.transpose.int(x, 0, 1).contiguous()],
        "outputs": "float32;n=%d" % (ROWS * COLS),
        "grid": ((COLS + 15) // 16, (ROWS + 15) // 16),
        "block": (16, 16),
    }

def run(inputs, kernel):
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(ROWS), np.uint32(COLS),
    ])]
