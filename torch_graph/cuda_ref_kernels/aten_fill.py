"""Reference for aten.fill — PyTorch implementation.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_fill.py --once
"""
import torch

def init_once():
    x = torch.randn(1024, device='cuda')
    return {"inputs": [x], "expected": [torch.ops.aten.fill.Scalar(x, 3.14)]}

def run(inputs):
    x = inputs[0]
    return [torch.ops.aten.fill.Scalar(x, 3.14)]
