"""Reference CUDA kernel for aten.nll_loss_forward — negative log likelihood.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_nll_loss_forward.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_nll_loss_kernel(
    const float *log_probs, const long *target, float *output,
    unsigned int N, unsigned int C
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float sum = 0.0f;
    for (unsigned int i = 0; i < N; i++) {
        long t = target[i];
        sum -= log_probs[i * C + t];
    }
    output[0] = sum / (float)N;
}
"""

NN, CC = 16, 10

def init_once():
    log_probs = torch.randn(NN, CC, device="cuda").log_softmax(dim=-1)
    target = torch.randint(0, CC, (NN,), device="cuda")
    return {
        "kernel_source": KERNEL_SRC, "inputs": [log_probs, target],
        "expected": [torch.ops.aten.nll_loss_forward.default(log_probs, target, None, 1, -100)[0].reshape(1)],
        "outputs": ["float32;n=1"], "grid": ((1 + 255) // 256,),
        "grid": (1,),
        "block": (1,), "atol": 1e-4,
    }

def run(inputs, kernel):
    log_probs, target = inputs
    return [kernel(log_probs, target, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.out_ptr(0),
        np.uint32(NN), np.uint32(CC),
    ])]
