"""Reference for aten.permute — PyTorch implementation.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_permute.py --once
"""
import torch

def init_once():
    x = torch.randn(4, 8, 16, device='cuda')
    return {"inputs": [x], "expected": [torch.ops.aten.permute.default(x, [2, 0, 1]).contiguous()]}

def run(inputs):
    x = inputs[0]
    return [torch.ops.aten.permute.default(x, [2, 0, 1]).contiguous()]
