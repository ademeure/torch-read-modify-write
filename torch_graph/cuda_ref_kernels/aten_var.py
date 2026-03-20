"""Reference for aten.var — PyTorch implementation.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_var.py --once
"""
import torch

def init_once():
    x = torch.randn(32, 64, device='cuda')
    return {"inputs": [x], "expected": [torch.ops.aten.var.correction(x, [-1])]}

def run(inputs):
    x = inputs[0]
    return [torch.ops.aten.var.correction(x, [-1])]
