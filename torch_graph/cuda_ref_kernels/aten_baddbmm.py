"""Reference for aten.baddbmm — PyTorch implementation.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_baddbmm.py --once
"""
import torch

def init_once():
    self = torch.randn(4, 16, 24, device='cuda')
    A = torch.randn(4, 16, 32, device='cuda')
    B = torch.randn(4, 32, 24, device='cuda')
    return {"inputs": [self, A, B], "expected": [torch.ops.aten.baddbmm.default(self, A, B)]}

def run(inputs):
    self = inputs[0]
    A = inputs[1]
    B = inputs[2]
    return [torch.ops.aten.baddbmm.default(self, A, B)]
