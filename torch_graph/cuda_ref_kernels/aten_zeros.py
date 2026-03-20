"""Reference for aten.zeros — PyTorch implementation.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_zeros.py --once
"""
import torch

def init_once():
    shape = (32, 64)
    return {"inputs": [shape], "expected": [torch.zeros(*shape, device='cuda')]}

def run(inputs):
    shape = inputs[0]
    return [torch.zeros(*shape, device='cuda')]
