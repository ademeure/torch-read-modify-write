"""Reference for aten.contiguous — PyTorch implementation.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_contiguous.py --once
"""
import torch

def init_once():
    x = torch.randn(32, 64, device='cuda').t()
    return {"inputs": [x], "expected": [x.contiguous()]}

def run(inputs):
    x = inputs[0]
    return [x.contiguous()]
