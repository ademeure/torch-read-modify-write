"""Reference CUDA kernel for aten.log_sigmoid.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_log_sigmoid.py --once
"""
import torch

KERNEL_SRC = r"""
extern "C" __global__ void aten_log_sigmoid_forward(const float *in0, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in0[i];
        out0[i] = (-logf(1.0f + expf(-x)));
    }
}
"""

def init_once():
    x = torch.randn(1024, device='cuda')
    return {
        "kernel_source": KERNEL_SRC,
        "inputs": [x],
        "expected": [torch.ops.aten.log_sigmoid_forward.default(x)[0]],
        "atol": 1e-05,
    }

def run(inputs, kernel):
    return [kernel(*inputs)]
