"""Reference for aten.dot — PyTorch implementation.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_dot.py --once
"""
import torch

def init_once():
    a = torch.randn(256, device='cuda')
    b = torch.randn(256, device='cuda')
    return {"inputs": [a, b], "expected": [torch.ops.aten.dot.default(a, b)]}

def run(inputs):
    a = inputs[0]
    b = inputs[1]
    return [torch.ops.aten.dot.default(a, b)]
