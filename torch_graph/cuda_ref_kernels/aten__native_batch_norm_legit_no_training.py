"""Reference CUDA kernel for aten._native_batch_norm_legit_no_training."""
import torch
import numpy as np
from torch_graph.cuda_ref_kernels.aten_native_batch_norm import KERNEL_SRC

NN, CC, HW = 2, 8, 16

def init_once():
    x = torch.randn(NN, CC, 4, 4, device="cuda")
    w, b = torch.randn(CC, device="cuda"), torch.randn(CC, device="cuda")
    rm, rv = torch.randn(CC, device="cuda"), torch.rand(CC, device="cuda") + 0.1
    total = x.numel()
    return {"kernel_source": KERNEL_SRC, "inputs": [x.contiguous(), w, b, rm, rv],
            "expected": [torch.ops.aten._native_batch_norm_legit_no_training.default(x, w, b, rm, rv, 0.1, 1e-5)[0].flatten()],
            "outputs": ["float32;n=%d" % total], "grid": ((total + 255) // 256,), "atol": 1e-4}

def run(inputs, kernel):
    total = inputs[0].numel()
    return [kernel(*inputs, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.in_ptr(2), kernel.in_ptr(3), kernel.in_ptr(4),
        kernel.out_ptr(0), np.uint32(NN), np.uint32(CC), np.uint32(HW), np.float32(1e-5),
    ])]
