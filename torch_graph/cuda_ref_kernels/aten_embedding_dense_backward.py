"""Reference for aten.embedding_dense_backward — gradient of embedding lookup."""
import torch

def init_once():
    grad = torch.randn(32, 64, device="cuda")
    indices = torch.randint(0, 100, (32,), device="cuda")
    return {"inputs": [grad, indices],
            "expected": [torch.ops.aten.embedding_dense_backward.default(grad, indices, 100, -1, False)]}

def run(inputs):
    return [torch.ops.aten.embedding_dense_backward.default(inputs[0], inputs[1], 100, -1, False)]
