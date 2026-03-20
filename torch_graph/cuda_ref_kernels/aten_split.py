"""Reference for aten.split — PyTorch implementation.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_split.py --once
"""
import torch

def init_once():
    x = torch.randn(32, 64, device='cuda')
    return {"inputs": [x], "expected": [list(torch.ops.aten.split.Tensor(x, 8, 0))[0].contiguous()]}

def run(inputs):
    x = inputs[0]
    return [list(torch.ops.aten.split.Tensor(x, 8, 0))[0].contiguous()]
