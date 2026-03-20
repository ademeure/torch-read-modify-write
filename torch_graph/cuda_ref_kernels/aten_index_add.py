"""Reference for aten.index_add — PyTorch implementation.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_index_add.py --once
"""
import torch

def init_once():
    x = torch.zeros(32, 64, device='cuda')
    idx = torch.tensor([0, 5, 10, 15, 20], device='cuda')
    src = torch.randn(5, 64, device='cuda')
    return {"inputs": [x, idx, src], "expected": [torch.ops.aten.index_add.default(x, 0, idx, src)]}

def run(inputs):
    x = inputs[0]
    idx = inputs[1]
    src = inputs[2]
    return [torch.ops.aten.index_add.default(x, 0, idx, src)]
