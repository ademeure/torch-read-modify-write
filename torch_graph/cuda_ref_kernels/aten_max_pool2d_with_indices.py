"""Reference for aten.max_pool2d_with_indices."""
import torch

def init_once():
    x = torch.randn(1, 4, 8, 8, device="cuda")
    return {"inputs": [x], "expected": [torch.ops.aten.max_pool2d_with_indices.default(x, [2,2], [2,2])[0]]}

def run(inputs):
    return [torch.ops.aten.max_pool2d_with_indices.default(inputs[0], [2,2], [2,2])[0]]
