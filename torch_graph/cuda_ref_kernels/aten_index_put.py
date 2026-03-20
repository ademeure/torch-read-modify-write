"""Reference for aten.index_put — advanced index assignment."""
import torch

def init_once():
    x = torch.randn(32, 64, device="cuda")
    idx = torch.tensor([0, 5, 10, 15], device="cuda")
    vals = torch.randn(4, 64, device="cuda")
    return {"inputs": [x, idx, vals], "expected": [torch.ops.aten.index_put.default(x, [idx], vals)]}

def run(inputs):
    return [torch.ops.aten.index_put.default(inputs[0], [inputs[1]], inputs[2])]
