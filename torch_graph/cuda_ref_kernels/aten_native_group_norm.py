"""Reference for aten.native_group_norm — PyTorch implementation.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_native_group_norm.py --once
"""
import torch

def init_once():
    x = torch.randn(2, 8, 4, 4, device='cuda')
    w = torch.randn(8, device='cuda')
    b = torch.randn(8, device='cuda')
    return {"inputs": [x, w, b], "expected": [torch.ops.aten.native_group_norm.default(x, w, b, 2, 8, 16, 4, 1e-5)[0]]}

def run(inputs):
    x = inputs[0]
    w = inputs[1]
    b = inputs[2]
    return [torch.ops.aten.native_group_norm.default(x, w, b, 2, 8, 16, 4, 1e-5)[0]]
