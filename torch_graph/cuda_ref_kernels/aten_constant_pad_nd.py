"""Reference for aten.constant_pad_nd — PyTorch implementation.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_constant_pad_nd.py --once
"""
import torch

def init_once():
    x = torch.randn(2, 8, 8, device='cuda')
    return {"inputs": [x], "expected": [torch.ops.aten.constant_pad_nd.default(x, [1, 1, 1, 1], 0.0)]}

def run(inputs):
    x = inputs[0]
    return [torch.ops.aten.constant_pad_nd.default(x, [1, 1, 1, 1], 0.0)]
