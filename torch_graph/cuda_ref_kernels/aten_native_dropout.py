"""Reference for aten.native_dropout — PyTorch implementation.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_native_dropout.py --once
"""
import torch

def init_once():
    x = torch.randn(1024, device='cuda')
    mask = (torch.rand(1024, device='cuda') > 0.5).float()
    return {"inputs": [x, mask], "expected": [x * mask * 2.0]}

def run(inputs):
    x = inputs[0]
    mask = inputs[1]
    return [x * mask * 2.0]
