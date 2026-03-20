"""Reference for aten._log_softmax — PyTorch implementation.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten__log_softmax.py --once
"""
import torch

def init_once():
    x = torch.randn(8, 64, device='cuda')
    return {"inputs": [x], "expected": [torch.ops.aten._log_softmax.default(x, -1, False)]}

def run(inputs):
    x = inputs[0]
    return [torch.ops.aten._log_softmax.default(x, -1, False)]
