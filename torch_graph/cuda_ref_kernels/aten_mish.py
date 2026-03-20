"""Reference CUDA kernel for aten.mish.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_mish.py --once
"""
import torch

KERNEL_SRC = r"""
extern "C" __global__ void aten_mish(const float *in0, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in0[i];
        out0[i] = (x * tanhf((x > 20.0f) ? x : logf(1.0f + expf(x))));
    }
}
"""

def init_once():
    x = torch.randn(1024, device='cuda')
    return {
        "kernel_source": KERNEL_SRC,
        "inputs": [x],
        "expected": [torch.ops.aten.mish.default(x)],
        "atol": 0.0001,
    }

def run(inputs, kernel):
    return [kernel(*inputs)]
