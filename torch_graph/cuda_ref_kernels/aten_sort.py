"""Reference CUDA kernel for aten.sort — bubble sort reference.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_sort.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_sort_kernel(
    const float *input, float *values,
    unsigned int rows, unsigned int cols
) {
    unsigned int row = blockIdx.x;
    if (row >= rows) return;
    const float *ri = input + row * cols;
    float *rv = values + row * cols;
    for (unsigned int j = 0; j < cols; j++) rv[j] = ri[j];
    for (unsigned int i = 0; i < cols; i++) {
        for (unsigned int j = i + 1; j < cols; j++) {
            if (rv[j] < rv[i]) {
                float tmp = rv[i]; rv[i] = rv[j]; rv[j] = tmp;
            }
        }
    }
}
"""

ROWS, COLS = 8, 32

def init_once():
    x = torch.randn(ROWS, COLS, device="cuda")
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x],
        "expected": [torch.ops.aten.sort.default(x, -1)[0].flatten()],
        "outputs": ["float32;n=%d" % (ROWS * COLS)],
        "grid": (ROWS,),
        "block": (1,),
    }

def run(inputs, kernel):
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(ROWS), np.uint32(COLS),
    ])]
