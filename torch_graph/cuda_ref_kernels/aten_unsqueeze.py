"""Reference for aten.unsqueeze — PyTorch implementation.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_unsqueeze.py --once
"""
import torch

def init_once():
    x = torch.randn(32, 64, device='cuda')
    return {"inputs": [x], "expected": [torch.ops.aten.unsqueeze.default(x, 0).contiguous()]}

def run(inputs):
    x = inputs[0]
    return [torch.ops.aten.unsqueeze.default(x, 0).contiguous()]
