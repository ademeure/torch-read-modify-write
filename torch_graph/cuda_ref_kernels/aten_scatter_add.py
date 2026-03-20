"""Reference for aten.scatter_add."""
import torch

def init_once():
    x = torch.zeros(8, 32, device="cuda")
    idx = torch.randint(0, 32, (8, 16), device="cuda")
    src = torch.randn(8, 16, device="cuda")
    return {"inputs": [x, idx, src], "expected": [torch.ops.aten.scatter_add.default(x, 1, idx, src)]}

def run(inputs):
    return [torch.ops.aten.scatter_add.default(inputs[0], 1, inputs[1], inputs[2])]
