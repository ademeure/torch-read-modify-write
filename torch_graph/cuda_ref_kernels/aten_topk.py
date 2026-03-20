"""Reference for aten.topk — PyTorch implementation.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_topk.py --once
"""
import torch

def init_once():
    x = torch.randn(8, 32, device='cuda')
    return {"inputs": [x], "expected": [torch.ops.aten.topk.default(x, 5, -1)[0]]}

def run(inputs):
    x = inputs[0]
    return [torch.ops.aten.topk.default(x, 5, -1)[0]]
