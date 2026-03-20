"""Reference CUDA kernel for aten.argmax — index of maximum value.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_argmax.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_argmax_kernel(
    const float *input, float *output, unsigned int rows, unsigned int cols
) {
    unsigned int row = blockIdx.x;
    if (row >= rows) return;
    const float *ri = input + row * cols;
    float best = ri[0];
    float best_idx = 0.0f;
    for (unsigned int j = 1; j < cols; j++) {
        if (ri[j] > best) { best = ri[j]; best_idx = (float)j; }
    }
    output[row] = best_idx;
}
"""

ROWS, COLS = 32, 64

def init_once():
    x = torch.randn(ROWS, COLS, device="cuda")
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x],
        "expected": [torch.ops.aten.argmax.default(x, -1).float()],
        "outputs": ["float32;n=%d" % ROWS],
        "grid": (ROWS,),
        "block": (1,),
    }

def run(inputs, kernel):
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(ROWS), np.uint32(COLS),
    ])]
