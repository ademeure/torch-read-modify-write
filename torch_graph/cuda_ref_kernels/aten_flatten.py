"""Reference for aten.flatten — PyTorch implementation.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_flatten.py --once
"""
import torch

def init_once():
    x = torch.randn(4, 8, 16, device='cuda')
    return {"inputs": [x], "expected": [torch.ops.aten.flatten.using_ints(x, 1, 2).contiguous()]}

def run(inputs):
    x = inputs[0]
    return [torch.ops.aten.flatten.using_ints(x, 1, 2).contiguous()]
