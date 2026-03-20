"""Reference for aten.repeat — PyTorch implementation.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_repeat.py --once
"""
import torch

def init_once():
    x = torch.randn(8, 16, device='cuda')
    return {"inputs": [x], "expected": [torch.ops.aten.repeat.default(x, [3, 2]).contiguous()]}

def run(inputs):
    x = inputs[0]
    return [torch.ops.aten.repeat.default(x, [3, 2]).contiguous()]
