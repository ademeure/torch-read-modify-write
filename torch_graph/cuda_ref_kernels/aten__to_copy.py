"""Reference for aten._to_copy — PyTorch implementation.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten__to_copy.py --once
"""
import torch

def init_once():
    x = torch.randn(1024, device='cuda')
    return {"inputs": [x], "expected": [torch.ops.aten._to_copy.default(x)]}

def run(inputs):
    x = inputs[0]
    return [torch.ops.aten._to_copy.default(x)]
