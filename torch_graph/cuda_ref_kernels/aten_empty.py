"""Reference for aten.empty — allocate uninitialized tensor."""
import torch

def init_once():
    return {"inputs": [], "expected": [torch.empty(1024, device="cuda")]}

def run(inputs):
    return [torch.empty(1024, device="cuda")]
