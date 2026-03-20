"""Reference CUDA kernel for aten.split — extract first chunk along dim 0.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_split.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_split_copy(
    const float *input, float *out, unsigned int offset, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = input[offset + i];
}
"""

def init_once():
    x = torch.randn(32, 64, device="cuda")
    chunk = list(torch.ops.aten.split.Tensor(x, 8, 0))[0].contiguous()
    total = 8 * 64
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x],
        "expected": [chunk.flatten()],
        "outputs": ["float32;n=%d" % total],
        "grid": ((total + 255) // 256,),
    }

def run(inputs, kernel):
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(0), np.uint32(8 * 64),
    ])]
