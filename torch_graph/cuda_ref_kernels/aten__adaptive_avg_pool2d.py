"""Reference CUDA kernel for aten._adaptive_avg_pool2d (alias for adaptive_avg_pool2d)."""
import torch
import numpy as np
# Reuse the adaptive_avg_pool2d kernel
from torch_graph.cuda_ref_kernels.aten_adaptive_avg_pool2d import KERNEL_SRC, make_inputs, expected

def init_once():
    inputs = make_inputs() if callable(getattr(__import__('torch_graph.cuda_ref_kernels.aten_adaptive_avg_pool2d', fromlist=['make_inputs']), 'make_inputs', None)) else None
    if inputs is None:
        x = torch.randn(1, 4, 8, 8, device="cuda")
        inputs = [x.contiguous()]
    return {"kernel_source": KERNEL_SRC, "inputs": inputs, "expected": [torch.ops.aten._adaptive_avg_pool2d.default(inputs[0], [1, 1]).flatten()],
            "outputs": ["float32;n=%d" % (inputs[0].size(0) * inputs[0].size(1))],
            "grid": ((inputs[0].size(0) * inputs[0].size(1) + 255) // 256,)}

def run(inputs, kernel):
    return [kernel(*inputs)]
