"""Reference for aten.detach — detach from autograd."""
import torch

def init_once():
    x = torch.randn(1024, device="cuda", requires_grad=True)
    return {"inputs": [x], "expected": [x.detach()]}

def run(inputs):
    return [inputs[0].detach()]
