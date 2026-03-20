"""Reference for aten.full — PyTorch implementation.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_full.py --once
"""
import torch

def init_once():
    shape = (32, 64)
    return {"inputs": [shape], "expected": [torch.full(shape, 3.14, device='cuda')]}

def run(inputs):
    shape = inputs[0]
    return [torch.full(shape, 3.14, device='cuda')]
