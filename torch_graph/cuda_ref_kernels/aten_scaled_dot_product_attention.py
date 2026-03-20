"""Reference for aten.scaled_dot_product_attention — PyTorch implementation.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_scaled_dot_product_attention.py --once
"""
import torch

def init_once():
    Q = torch.randn(1, 2, 8, 16, device='cuda')
    K = torch.randn(1, 2, 8, 16, device='cuda')
    V = torch.randn(1, 2, 8, 16, device='cuda')
    return {"inputs": [Q, K, V], "expected": [torch.nn.functional.scaled_dot_product_attention(Q, K, V)]}

def run(inputs):
    Q = inputs[0]
    K = inputs[1]
    V = inputs[2]
    return [torch.nn.functional.scaled_dot_product_attention(Q, K, V)]
