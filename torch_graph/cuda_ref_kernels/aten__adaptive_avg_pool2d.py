"""Reference for aten._adaptive_avg_pool2d."""
import torch

def init_once():
    x = torch.randn(1, 4, 8, 8, device="cuda")
    return {"inputs": [x], "expected": [torch.ops.aten._adaptive_avg_pool2d.default(x, [1, 1])]}

def run(inputs):
    return [torch.ops.aten._adaptive_avg_pool2d.default(inputs[0], [1, 1])]
