"""Reference for aten.full_like."""
import torch

def init_once():
    x = torch.randn(1024, device="cuda")
    return {"inputs": [x], "expected": [torch.full_like(x, 3.14)]}

def run(inputs):
    return [torch.full_like(inputs[0], 3.14)]
