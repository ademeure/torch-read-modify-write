"""Reference CUDA kernel for aten.bitwise_or.Tensor."""
import torch

KERNEL_SRC = r"""
extern "C" __global__ void aten_bitwise_or(const int *in0, const int *in1, int *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out0[i] = in0[i] | in1[i];
}
"""

def init_once():
    a = torch.randint(-1000, 1000, (1024,), device="cuda", dtype=torch.int32)
    b = torch.randint(-1000, 1000, (1024,), device="cuda", dtype=torch.int32)
    return {"kernel_source": KERNEL_SRC, "inputs": [a, b],
            "expected": [torch.ops.aten.bitwise_or.Tensor(a, b)]}

def run(inputs, kernel):
    return [kernel(*inputs)]
