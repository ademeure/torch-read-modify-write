"""Reference for aten.ones — PyTorch implementation.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_ones.py --once
"""
import torch

def init_once():
    shape = (32, 64)
    return {"inputs": [shape], "expected": [torch.ones(*shape, device='cuda')]}

def run(inputs):
    shape = inputs[0]
    return [torch.ones(*shape, device='cuda')]
