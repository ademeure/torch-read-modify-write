"""Reference for aten.nonzero — indices of nonzero elements."""
import torch

def init_once():
    x = torch.tensor([0.0, 1.0, 0.0, -1.0, 0.0, 3.14, 0.0, 0.0] * 128, device="cuda")
    return {"inputs": [x], "expected": [torch.ops.aten.nonzero.default(x)]}

def run(inputs):
    return [torch.ops.aten.nonzero.default(inputs[0])]
