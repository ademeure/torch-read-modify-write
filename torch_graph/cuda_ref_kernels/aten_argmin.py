"""Reference for aten.argmin — PyTorch implementation.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_argmin.py --once
"""
import torch

def init_once():
    x = torch.randn(32, 64, device='cuda')
    return {"inputs": [x], "expected": [torch.ops.aten.argmin.default(x, -1)]}

def run(inputs):
    x = inputs[0]
    return [torch.ops.aten.argmin.default(x, -1)]
