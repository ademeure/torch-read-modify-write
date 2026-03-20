"""Reference CUDA kernel for aten.eye — identity matrix.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_eye.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_eye_kernel(float *output, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * n) return;
    unsigned int r = idx / n, c = idx % n;
    output[idx] = (r == c) ? 1.0f : 0.0f;
}
"""

N = 32

def init_once():
    return {
        "kernel_source": KERNEL_SRC, "inputs": [],
        "expected": [torch.eye(N, device='cuda').flatten()],
        "outputs": ["float32;n=%d" % (N * N)],
        "grid": (((N * N) + 255) // 256,),
    }

def run(inputs, kernel):
    return [kernel(params=[
        kernel.out_ptr(0), np.uint32(N),
    ])]
