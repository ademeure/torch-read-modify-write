"""Reference for aten.flip — PyTorch implementation.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_flip.py --once
"""
import torch

def init_once():
    x = torch.randn(16, 32, device='cuda')
    return {"inputs": [x], "expected": [torch.ops.aten.flip.default(x, [-1]).contiguous()]}

def run(inputs):
    x = inputs[0]
    return [torch.ops.aten.flip.default(x, [-1]).contiguous()]
