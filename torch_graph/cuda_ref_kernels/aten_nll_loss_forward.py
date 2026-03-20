"""Reference for aten.nll_loss_forward — PyTorch implementation.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_nll_loss_forward.py --once
"""
import torch

def init_once():
    log_probs = torch.randn(16, 10, device='cuda').log_softmax(dim=-1)
    target = torch.randint(0, 10, (16,), device='cuda')
    return {"inputs": [log_probs, target], "expected": [torch.ops.aten.nll_loss_forward.default(log_probs, target, None, 1, -100)[0]]}

def run(inputs):
    log_probs = inputs[0]
    target = inputs[1]
    return [torch.ops.aten.nll_loss_forward.default(log_probs, target, None, 1, -100)[0]]
