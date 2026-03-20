"""Reference for aten.native_batch_norm — PyTorch implementation.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_native_batch_norm.py --once
"""
import torch

def init_once():
    x = torch.randn(2, 8, 4, 4, device='cuda')
    w = torch.randn(8, device='cuda')
    b = torch.randn(8, device='cuda')
    rm = torch.randn(8, device='cuda')
    rv = torch.rand(8, device='cuda') + 0.1
    return {"inputs": [x, w, b, rm, rv], "expected": [torch.ops.aten.native_batch_norm.default(x, w, b, rm, rv, False, 0.1, 1e-5)[0]]}

def run(inputs):
    x = inputs[0]
    w = inputs[1]
    b = inputs[2]
    rm = inputs[3]
    rv = inputs[4]
    return [torch.ops.aten.native_batch_norm.default(x, w, b, rm, rv, False, 0.1, 1e-5)[0]]
