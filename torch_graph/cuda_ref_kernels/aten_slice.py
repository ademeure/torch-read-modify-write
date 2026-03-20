"""Reference for aten.slice — PyTorch implementation.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_slice.py --once
"""
import torch

def init_once():
    x = torch.randn(32, 64, device='cuda')
    return {"inputs": [x], "expected": [torch.ops.aten.slice.Tensor(x, 0, 4, 20).contiguous()]}

def run(inputs):
    x = inputs[0]
    return [torch.ops.aten.slice.Tensor(x, 0, 4, 20).contiguous()]
