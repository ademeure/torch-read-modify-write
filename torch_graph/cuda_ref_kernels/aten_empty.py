"""Reference for aten.empty (output undefined, verified via zeros)."""
import torch

def init_once():
    return {"inputs": [], "expected": [torch.zeros(1024, device="cuda")]}

def run(inputs):
    return [torch.zeros(1024, device="cuda")]
