"""Reference for aten.as_strided — general strided view."""
import torch

def init_once():
    x = torch.randn(64, device="cuda")
    return {"inputs": [x], "expected": [torch.ops.aten.as_strided.default(x, [8, 8], [8, 1]).contiguous()]}

def run(inputs):
    return [torch.ops.aten.as_strided.default(inputs[0], [8, 8], [8, 1]).contiguous()]
