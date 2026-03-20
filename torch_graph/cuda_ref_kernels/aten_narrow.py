"""Reference for aten.narrow — PyTorch implementation.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_narrow.py --once
"""
import torch

def init_once():
    x = torch.randn(32, 64, device='cuda')
    return {"inputs": [x], "expected": [torch.ops.aten.narrow.default(x, 0, 4, 10).contiguous()]}

def run(inputs):
    x = inputs[0]
    return [torch.ops.aten.narrow.default(x, 0, 4, 10).contiguous()]
