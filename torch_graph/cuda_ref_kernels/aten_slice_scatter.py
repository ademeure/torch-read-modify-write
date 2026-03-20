"""Reference for aten.slice_scatter."""
import torch

def init_once():
    x = torch.randn(8, 16, device="cuda")
    src = torch.randn(4, 16, device="cuda")
    return {"inputs": [x, src], "expected": [torch.ops.aten.slice_scatter.default(x, src, 0, 2, 6)]}

def run(inputs):
    return [torch.ops.aten.slice_scatter.default(inputs[0], inputs[1], 0, 2, 6)]
