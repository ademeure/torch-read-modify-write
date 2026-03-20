"""Reference for aten.gather — PyTorch implementation.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_gather.py --once
"""
import torch

def init_once():
    x = torch.randn(8, 32, device='cuda')
    idx = torch.randint(0, 32, (8, 16), device='cuda')
    return {"inputs": [x, idx], "expected": [torch.ops.aten.gather.default(x, 1, idx)]}

def run(inputs):
    x = inputs[0]
    idx = inputs[1]
    return [torch.ops.aten.gather.default(x, 1, idx)]
