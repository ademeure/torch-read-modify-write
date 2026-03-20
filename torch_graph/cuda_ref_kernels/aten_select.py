"""Reference for aten.select — PyTorch implementation.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_select.py --once
"""
import torch

def init_once():
    x = torch.randn(32, 64, device='cuda')
    return {"inputs": [x], "expected": [torch.ops.aten.select.int(x, 0, 5).contiguous()]}

def run(inputs):
    x = inputs[0]
    return [torch.ops.aten.select.int(x, 0, 5).contiguous()]
