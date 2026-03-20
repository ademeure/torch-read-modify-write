"""Reference CUDA kernel for aten.pow.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_pow.py --once
"""
import torch

KERNEL_SRC = r"""
extern "C" __global__ void aten_pow(const float *in0, const float *in1, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float a = in0[i];
        float b = in1[i];
        out0[i] = powf(a, b);
    }
}
"""

def init_once():
    a = torch.rand(1024, device='cuda') + 0.1
    b = torch.rand(1024, device='cuda') * 3
    return {
        "kernel_source": KERNEL_SRC,
        "inputs": [a, b],
        "expected": [torch.ops.aten.pow.Tensor_Tensor(a, b)],
        "atol": 0.0001,
    }

def run(inputs, kernel):
    return [kernel(*inputs)]
