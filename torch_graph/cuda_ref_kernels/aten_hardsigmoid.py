"""Reference CUDA kernel for aten.hardsigmoid.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_hardsigmoid.py --once
"""
import torch

KERNEL_SRC = r"""
extern "C" __global__ void aten_hardsigmoid(const float *in0, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in0[i];
        out0[i] = fminf(fmaxf(x / 6.0f + 0.5f, 0.0f), 1.0f);
    }
}
"""

def init_once():
    x = torch.randn(1024, device='cuda') * 5
    return {
        "kernel_source": KERNEL_SRC,
        "inputs": [x],
        "expected": [torch.ops.aten.hardsigmoid.default(x)],
        "atol": 1e-05,
    }

def run(inputs, kernel):
    return [kernel(*inputs)]
