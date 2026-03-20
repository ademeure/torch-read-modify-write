"""Reference for aten.index.Tensor — advanced indexing."""
import torch

def init_once():
    x = torch.randn(32, 64, device="cuda")
    idx = torch.randint(0, 32, (10,), device="cuda")
    return {"inputs": [x, idx], "expected": [torch.ops.aten.index.Tensor(x, [idx])]}

def run(inputs):
    return [torch.ops.aten.index.Tensor(inputs[0], [inputs[1]])]
