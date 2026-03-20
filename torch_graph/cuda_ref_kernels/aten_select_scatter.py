"""Reference for aten.select_scatter."""
import torch

def init_once():
    x = torch.randn(8, 16, device="cuda")
    src = torch.randn(16, device="cuda")
    return {"inputs": [x, src], "expected": [torch.ops.aten.select_scatter.default(x, src, 0, 3)]}

def run(inputs):
    return [torch.ops.aten.select_scatter.default(inputs[0], inputs[1], 0, 3)]
