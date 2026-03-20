"""Reference CUDA kernel for aten.topk — find k largest values.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_topk.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_topk_kernel(
    const float *input, float *values,
    unsigned int rows, unsigned int cols, unsigned int k
) {
    unsigned int row = blockIdx.x;
    if (row >= rows) return;
    const float *ri = input + row * cols;
    float *rv = values + row * k;
    // Track selected indices locally
    int selected[32];  // max k=32
    for (unsigned int i = 0; i < k; i++) {
        float best = -1e38f;
        int best_j = 0;
        for (unsigned int j = 0; j < cols; j++) {
            float v = ri[j];
            int already = 0;
            for (unsigned int p = 0; p < i; p++) {
                if (selected[p] == (int)j) { already = 1; break; }
            }
            if (!already && v > best) { best = v; best_j = j; }
        }
        rv[i] = best;
        selected[i] = best_j;
    }
}
"""

ROWS, COLS, K = 8, 32, 5

def init_once():
    x = torch.randn(ROWS, COLS, device="cuda")
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x],
        "expected": [torch.ops.aten.topk.default(x, K, -1)[0].flatten()],
        "outputs": ["float32;n=%d" % (ROWS * K)],
        "grid": (ROWS,),
        "block": (1,),
    }

def run(inputs, kernel):
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(ROWS), np.uint32(COLS), np.uint32(K),
    ])]
