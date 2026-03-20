"""Reference for aten.linspace — PyTorch implementation.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_linspace.py --once
"""
import torch

def init_once():
    start, end, steps = 0.0, 1.0, 100
    return {"inputs": [], "expected": [torch.linspace(start, end, steps, device='cuda')]}

def run(inputs):
    start, end, steps = 0.0, 1.0, 100
    return [torch.linspace(start, end, steps, device='cuda')]
