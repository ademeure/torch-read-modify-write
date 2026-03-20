"""Reference for aten.roll — PyTorch implementation.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_roll.py --once
"""
import torch

def init_once():
    x = torch.randn(256, device='cuda')
    return {"inputs": [x], "expected": [torch.ops.aten.roll.default(x, [10]).contiguous()]}

def run(inputs):
    x = inputs[0]
    return [torch.ops.aten.roll.default(x, [10]).contiguous()]
