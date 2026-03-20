"""Reference CUDA kernel for aten.cumsum — cumulative sum along last dim.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_cumsum.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_cumsum_kernel(
    const float *input, float *output, unsigned int rows, unsigned int cols
) {
    unsigned int row = blockIdx.x;
    if (row >= rows) return;
    const float *ri = input + row * cols;
    float *ro = output + row * cols;
    float acc = 0.0f;
    for (unsigned int j = 0; j < cols; j++) {
        acc += ri[j];
        ro[j] = acc;
    }
}
"""

ROWS, COLS = 8, 64

def init_once():
    x = torch.randn(ROWS, COLS, device="cuda")
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x],
        "expected": [torch.ops.aten.cumsum.default(x, -1).flatten()],
        "outputs": ["float32;n=%d" % (ROWS * COLS)],
        "grid": (ROWS,),
        "block": (1,), "atol": 1e-4,
    }

def run(inputs, kernel):
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(ROWS), np.uint32(COLS),
    ])]
