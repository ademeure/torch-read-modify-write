"""Reference CUDA kernel for aten.any — reduce: any nonzero along last dim."""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_any(
    const float *input, float *output, unsigned int rows, unsigned int cols
) {
    unsigned int row = blockIdx.x;
    if (row >= rows) return;
    const float *ri = input + row * cols;
    float found = 0.0f;
    for (unsigned int j = 0; j < cols; j++) {
        if (ri[j] != 0.0f) { found = 1.0f; break; }
    }
    output[row] = found;
}
"""

def init_once():
    x = torch.randn(32, 64, device="cuda")
    rows, cols = x.shape
    return {"kernel_source": KERNEL_SRC, "inputs": [x],
            "expected": [torch.ops.aten.any.dim(x, -1).float()],
            "outputs": ["float32;n=%d" % rows], "grid": (rows,), "block": (1,)}

def run(inputs, kernel):
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(inputs[0].size(0)), np.uint32(inputs[0].size(1)),
    ])]
