"""Reference for aten.eye — PyTorch implementation.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_eye.py --once
"""
import torch

def init_once():
    n = 32
    return {"inputs": [n], "expected": [torch.eye(n, device='cuda')]}

def run(inputs):
    n = inputs[0]
    return [torch.eye(n, device='cuda')]
