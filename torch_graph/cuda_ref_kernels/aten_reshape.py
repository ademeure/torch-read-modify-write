"""Reference for aten.reshape — PyTorch implementation.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_reshape.py --once
"""
import torch

def init_once():
    x = torch.randn(32, 64, device='cuda')
    return {"inputs": [x], "expected": [torch.ops.aten.reshape.default(x, [64, 32]).contiguous()]}

def run(inputs):
    x = inputs[0]
    return [torch.ops.aten.reshape.default(x, [64, 32]).contiguous()]
