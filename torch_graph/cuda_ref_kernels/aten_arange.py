"""Reference for aten.arange — PyTorch implementation.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_arange.py --once
"""
import torch

def init_once():
    end = 100
    return {"inputs": [end], "expected": [torch.ops.aten.arange.start_step(0, end, 1, dtype=torch.float32, device='cuda')]}

def run(inputs):
    end = inputs[0]
    return [torch.ops.aten.arange.start_step(0, end, 1, dtype=torch.float32, device='cuda')]
