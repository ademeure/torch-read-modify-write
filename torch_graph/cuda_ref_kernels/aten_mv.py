"""Reference for aten.mv — PyTorch implementation.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_mv.py --once
"""
import torch

def init_once():
    A = torch.randn(64, 32, device='cuda')
    x = torch.randn(32, device='cuda')
    return {"inputs": [A, x], "expected": [torch.ops.aten.mv.default(A, x)]}

def run(inputs):
    A = inputs[0]
    x = inputs[1]
    return [torch.ops.aten.mv.default(A, x)]
