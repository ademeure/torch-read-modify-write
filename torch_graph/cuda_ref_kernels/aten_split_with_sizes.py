"""Reference for aten.split_with_sizes — split tensor into chunks."""
import torch

def init_once():
    x = torch.randn(32, 64, device="cuda")
    return {"inputs": [x], "expected": [torch.ops.aten.split_with_sizes.default(x, [8, 8, 16], 0)[0].contiguous()]}

def run(inputs):
    return [torch.ops.aten.split_with_sizes.default(inputs[0], [8, 8, 16], 0)[0].contiguous()]
