"""Reference CUDA kernel for aten.dot — inner product of two vectors.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_dot.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_dot(
    const float *a, const float *b, float *out, unsigned int n
) {
    __shared__ float sdata[256];
    unsigned int tid = threadIdx.x;
    float v = 0.0f;
    for (unsigned int i = tid; i < n; i += blockDim.x)
        v += a[i] * b[i];
    sdata[tid] = v;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[0] = sdata[0];
}
"""

def init_once():
    a = torch.randn(256, device="cuda")
    b = torch.randn(256, device="cuda")
    return {
        "kernel_source": KERNEL_SRC, "inputs": [a, b],
        "expected": [torch.ops.aten.dot.default(a, b).reshape(1)],
        "outputs": ["float32;n=1"], "grid": ((1 + 255) // 256,),
        "grid": (1,),
        "block": (256,), "atol": 1e-3,
    }

def run(inputs, kernel):
    a, b = inputs
    return [kernel(a, b, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.out_ptr(0),
        np.uint32(a.numel()),
    ])]
