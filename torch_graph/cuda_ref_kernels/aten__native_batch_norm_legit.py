"""Reference for aten._native_batch_norm_legit."""
import torch

def init_once():
    x = torch.randn(2, 8, 4, 4, device="cuda")
    w, b = torch.randn(8, device="cuda"), torch.randn(8, device="cuda")
    rm, rv = torch.randn(8, device="cuda"), torch.rand(8, device="cuda") + 0.1
    return {"inputs": [x, w, b, rm, rv], "atol": 1e-4,
            "expected": [torch.ops.aten._native_batch_norm_legit.default(x, w, b, rm, rv, False, 0.1, 1e-5)[0]]}

def run(inputs):
    return [torch.ops.aten._native_batch_norm_legit.default(*inputs, False, 0.1, 1e-5)[0]]
