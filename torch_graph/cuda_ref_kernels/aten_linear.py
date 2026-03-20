"""Reference for aten.linear — PyTorch implementation.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_linear.py --once
"""
import torch

def init_once():
    x = torch.randn(32, 64, device='cuda')
    w = torch.randn(48, 64, device='cuda')
    b = torch.randn(48, device='cuda')
    return {"inputs": [x, w, b], "expected": [torch.ops.aten.linear.default(x, w, b)]}

def run(inputs):
    x = inputs[0]
    w = inputs[1]
    b = inputs[2]
    return [torch.ops.aten.linear.default(x, w, b)]
