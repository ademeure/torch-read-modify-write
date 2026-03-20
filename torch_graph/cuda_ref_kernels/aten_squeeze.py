"""Reference for aten.squeeze — PyTorch implementation.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_squeeze.py --once
"""
import torch

def init_once():
    x = torch.randn(32, 1, 64, device='cuda')
    return {"inputs": [x], "expected": [torch.ops.aten.squeeze.dim(x, 1).contiguous()]}

def run(inputs):
    x = inputs[0]
    return [torch.ops.aten.squeeze.dim(x, 1).contiguous()]
