"""Reference for aten.clone — PyTorch implementation.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_clone.py --once
"""
import torch

def init_once():
    x = torch.randn(1024, device='cuda')
    return {"inputs": [x], "expected": [torch.ops.aten.clone.default(x)]}

def run(inputs):
    x = inputs[0]
    return [torch.ops.aten.clone.default(x)]
