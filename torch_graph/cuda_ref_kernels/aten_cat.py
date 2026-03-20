"""Reference for aten.cat — PyTorch implementation.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_cat.py --once
"""
import torch

def init_once():
    a = torch.randn(8, 32, device='cuda')
    b = torch.randn(16, 32, device='cuda')
    return {"inputs": [a, b], "expected": [torch.ops.aten.cat.default([a, b], 0)]}

def run(inputs):
    a = inputs[0]
    b = inputs[1]
    return [torch.ops.aten.cat.default([a, b], 0)]
