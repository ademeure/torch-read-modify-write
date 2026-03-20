"""Reference CUDA kernel for aten._log_softmax — log(softmax(x)).
Run: kbox iterate torch_graph/cuda_ref_kernels/aten__log_softmax.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_log_softmax(
    const float *input, float *output, unsigned int rows, unsigned int cols
) {
    __shared__ float sdata[256];
    unsigned int row = blockIdx.x, tid = threadIdx.x;
    if (row >= rows) return;
    const float *ri = input + row * cols;
    float *ro = output + row * cols;
    float v = -1e38f;
    for (unsigned int j = tid; j < cols; j += blockDim.x) v = fmaxf(v, ri[j]);
    sdata[tid] = v; __syncthreads();
    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid+s]); __syncthreads();
    }
    float row_max = sdata[0];
    float lsum = 0.0f;
    for (unsigned int j = tid; j < cols; j += blockDim.x)
        lsum += expf(ri[j] - row_max);
    sdata[tid] = lsum; __syncthreads();
    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid+s]; __syncthreads();
    }
    float log_sum = logf(sdata[0]);
    for (unsigned int j = tid; j < cols; j += blockDim.x)
        ro[j] = (ri[j] - row_max) - log_sum;
}
"""

ROWS, COLS = 8, 64

def init_once():
    x = torch.randn(ROWS, COLS, device="cuda")
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x],
        "expected": [torch.ops.aten._log_softmax.default(x, -1, False).flatten()],
        "outputs": ["float32;n=%d" % (ROWS * COLS)],
        "grid": (ROWS,),
        "block": (256,), "atol": 1e-5,
    }

def run(inputs, kernel):
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(ROWS), np.uint32(COLS),
    ])]
