"""Reference CUDA kernel for aten.flip — reverse along last dimension.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_flip.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_flip_2d(
    const float *input, float *output, unsigned int rows, unsigned int cols
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    unsigned int r = idx / cols, c = idx % cols;
    output[r * cols + (cols - 1 - c)] = input[idx];
}
"""

ROWS, COLS = 16, 32

def init_once():
    x = torch.randn(ROWS, COLS, device="cuda")
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x],
        "expected": [torch.ops.aten.flip.default(x, [-1]).contiguous().flatten()],
        "outputs": ["float32;n=%d" % (ROWS * COLS)],
        "grid": (((ROWS * COLS) + 255) // 256,),
    }

def run(inputs, kernel):
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(ROWS), np.uint32(COLS),
    ])]
