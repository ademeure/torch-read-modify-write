"""Reference for aten.convolution — PyTorch implementation.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_convolution.py --once
"""
import torch

def init_once():
    x = torch.randn(1, 3, 8, 8, device='cuda')
    w = torch.randn(16, 3, 3, 3, device='cuda')
    b = torch.randn(16, device='cuda')
    return {"inputs": [x, w, b], "expected": [torch.ops.aten.convolution.default(x, w, b, [1,1], [1,1], [1,1], False, [0,0], 1)]}

def run(inputs):
    x = inputs[0]
    w = inputs[1]
    b = inputs[2]
    return [torch.ops.aten.convolution.default(x, w, b, [1,1], [1,1], [1,1], False, [0,0], 1)]
