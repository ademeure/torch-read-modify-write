"""Reference for aten.cumprod — PyTorch implementation.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_cumprod.py --once
"""
import torch

def init_once():
    x = torch.rand(8, 16, device='cuda') + 0.5
    return {"inputs": [x], "expected": [torch.ops.aten.cumprod.default(x, -1)]}

def run(inputs):
    x = inputs[0]
    return [torch.ops.aten.cumprod.default(x, -1)]
