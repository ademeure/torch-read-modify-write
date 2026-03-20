"""Reference for aten.stack — PyTorch implementation.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_stack.py --once
"""
import torch

def init_once():
    a = torch.randn(64, device='cuda')
    b = torch.randn(64, device='cuda')
    return {"inputs": [a, b], "expected": [torch.ops.aten.stack.default([a, b], 0)]}

def run(inputs):
    a = inputs[0]
    b = inputs[1]
    return [torch.ops.aten.stack.default([a, b], 0)]
