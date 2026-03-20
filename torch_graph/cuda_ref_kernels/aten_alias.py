"""Reference for aten.alias — identity op."""
import torch

def init_once():
    x = torch.randn(1024, device="cuda")
    return {"inputs": [x], "expected": [torch.ops.aten.alias.default(x)]}

def run(inputs):
    return [inputs[0]]
