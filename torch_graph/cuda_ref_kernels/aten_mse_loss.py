"""Reference for aten.mse_loss — PyTorch implementation.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_mse_loss.py --once
"""
import torch

def init_once():
    x = torch.randn(256, device='cuda')
    y = torch.randn(256, device='cuda')
    return {"inputs": [x, y], "expected": [torch.ops.aten.mse_loss.default(x, y)]}

def run(inputs):
    x = inputs[0]
    y = inputs[1]
    return [torch.ops.aten.mse_loss.default(x, y)]
