"""Reference for aten.scatter — PyTorch implementation.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_scatter.py --once
"""
import torch

def init_once():
    x = torch.zeros(8, 32, device='cuda')
    idx = torch.randint(0, 32, (8, 16), device='cuda')
    src = torch.randn(8, 16, device='cuda')
    return {"inputs": [x, idx, src], "expected": [torch.ops.aten.scatter.src(x, 1, idx, src)]}

def run(inputs):
    x = inputs[0]
    idx = inputs[1]
    src = inputs[2]
    return [torch.ops.aten.scatter.src(x, 1, idx, src)]
