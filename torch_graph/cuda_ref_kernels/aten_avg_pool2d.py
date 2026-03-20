"""Reference for aten.avg_pool2d — PyTorch implementation.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_avg_pool2d.py --once
"""
import torch

def init_once():
    x = torch.randn(1, 4, 8, 8, device='cuda')
    return {"inputs": [x], "expected": [torch.ops.aten.avg_pool2d.default(x, [2,2], [2,2])]}

def run(inputs):
    x = inputs[0]
    return [torch.ops.aten.avg_pool2d.default(x, [2,2], [2,2])]
