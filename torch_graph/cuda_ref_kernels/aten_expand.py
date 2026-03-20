"""Reference for aten.expand — PyTorch implementation.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_expand.py --once
"""
import torch

def init_once():
    x = torch.randn(1, 64, device='cuda')
    return {"inputs": [x], "expected": [torch.ops.aten.expand.default(x, [32, 64]).contiguous()]}

def run(inputs):
    x = inputs[0]
    return [torch.ops.aten.expand.default(x, [32, 64]).contiguous()]
