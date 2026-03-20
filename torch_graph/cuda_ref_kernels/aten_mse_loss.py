"""Reference CUDA kernel for aten.mse_loss — mean squared error.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_mse_loss.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_mse_kernel(
    const float *input, const float *target, float *output, unsigned int n
) {
    __shared__ float sdata[256];
    unsigned int tid = threadIdx.x;
    float v = 0.0f;
    for (unsigned int i = tid; i < n; i += blockDim.x) {
        float d = input[i] - target[i];
        v += d * d;
    }
    sdata[tid] = v; __syncthreads();
    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid+s]; __syncthreads();
    }
    if (tid == 0) output[0] = sdata[0] / (float)n;
}
"""

N = 256

def init_once():
    x = torch.randn(N, device="cuda")
    y = torch.randn(N, device="cuda")
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x, y],
        "expected": [torch.ops.aten.mse_loss.default(x, y).reshape(1)],
        "outputs": ["float32;n=1"],
        "grid": (1,),
        "block": (256,), "atol": 1e-4,
    }

def run(inputs, kernel):
    x, y = inputs
    return [kernel(x, y, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.out_ptr(0),
        np.uint32(x.numel()),
    ])]
