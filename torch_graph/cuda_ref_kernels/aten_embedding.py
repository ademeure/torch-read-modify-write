"""Reference for aten.embedding — PyTorch implementation.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_embedding.py --once
"""
import torch

def init_once():
    weight = torch.randn(100, 64, device='cuda')
    indices = torch.randint(0, 100, (32,), device='cuda')
    return {"inputs": [weight, indices], "expected": [torch.ops.aten.embedding.default(weight, indices)]}

def run(inputs):
    weight = inputs[0]
    indices = inputs[1]
    return [torch.ops.aten.embedding.default(weight, indices)]
