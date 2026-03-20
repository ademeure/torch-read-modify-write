"""Reference for aten.masked_scatter."""
import torch

def init_once():
    x = torch.randn(8, 16, device="cuda")
    mask = torch.randint(0, 2, (8, 16), device="cuda").bool()
    source = torch.randn(mask.sum().item(), device="cuda")
    return {"inputs": [x, mask, source],
            "expected": [torch.ops.aten.masked_scatter.default(x, mask, source)]}

def run(inputs):
    return [torch.ops.aten.masked_scatter.default(inputs[0], inputs[1], inputs[2])]
