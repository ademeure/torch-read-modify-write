"""Reference CUDA kernel for aten.repeat — tile tensor along dimensions.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_repeat.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_repeat_2d(
    const float *input, float *output,
    unsigned int R, unsigned int C, unsigned int rr, unsigned int rc
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int out_rows = R * rr, out_cols = C * rc;
    if (idx >= out_rows * out_cols) return;
    unsigned int r = idx / out_cols, c = idx % out_cols;
    output[idx] = input[(r % R) * C + (c % C)];
}
"""

R, C, RR, RC = 8, 16, 3, 2

def init_once():
    x = torch.randn(R, C, device="cuda")
    total = R * RR * C * RC
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x],
        "expected": [torch.ops.aten.repeat.default(x, [RR, RC]).contiguous().flatten()],
        "outputs": ["float32;n=%d" % total],
        "grid": ((total + 255) // 256,),
    }

def run(inputs, kernel):
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(R), np.uint32(C), np.uint32(RR), np.uint32(RC),
    ])]
