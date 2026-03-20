"""Reference for aten.index_select — PyTorch implementation.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_index_select.py --once
"""
import torch

def init_once():
    x = torch.randn(32, 64, device='cuda')
    idx = torch.tensor([0, 5, 10, 15, 31], device='cuda')
    return {"inputs": [x, idx], "expected": [torch.ops.aten.index_select.default(x, 0, idx)]}

def run(inputs):
    x = inputs[0]
    idx = inputs[1]
    return [torch.ops.aten.index_select.default(x, 0, idx)]
