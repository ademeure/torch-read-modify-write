"""Reference for aten.cumsum — PyTorch implementation.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_cumsum.py --once
"""
import torch

def init_once():
    x = torch.randn(8, 64, device='cuda')
    return {"inputs": [x], "expected": [torch.ops.aten.cumsum.default(x, -1)]}

def run(inputs):
    x = inputs[0]
    return [torch.ops.aten.cumsum.default(x, -1)]
