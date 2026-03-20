"""Reference for aten.tril — PyTorch implementation.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_tril.py --once
"""
import torch

def init_once():
    x = torch.randn(16, 16, device='cuda')
    return {"inputs": [x], "expected": [torch.ops.aten.tril.default(x)]}

def run(inputs):
    x = inputs[0]
    return [torch.ops.aten.tril.default(x)]
