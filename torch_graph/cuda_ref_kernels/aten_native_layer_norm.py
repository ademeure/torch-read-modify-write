"""Reference for aten.native_layer_norm — PyTorch implementation.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_native_layer_norm.py --once
"""
import torch

def init_once():
    x = torch.randn(8, 64, device='cuda')
    w = torch.randn(64, device='cuda')
    b = torch.randn(64, device='cuda')
    return {"inputs": [x, w, b], "expected": [torch.ops.aten.native_layer_norm.default(x, [64], w, b, 1e-5)[0]]}

def run(inputs):
    x = inputs[0]
    w = inputs[1]
    b = inputs[2]
    return [torch.ops.aten.native_layer_norm.default(x, [64], w, b, 1e-5)[0]]
