"""Reference CUDA kernel for aten._native_batch_norm_legit (eval mode with running stats)."""
import torch
from torch_graph.cuda_ref_kernels.aten_native_batch_norm import KERNEL_SRC

def init_once():
    x = torch.randn(2, 8, 4, 4, device="cuda")
    w = torch.randn(8, device="cuda")
    b = torch.randn(8, device="cuda")
    rm = torch.randn(8, device="cuda")
    rv = torch.rand(8, device="cuda") + 0.1
    total = x.numel()
    return {"kernel_source": KERNEL_SRC, "inputs": [x.contiguous(), w, b, rm, rv],
            "expected": [torch.ops.aten._native_batch_norm_legit.default(x, w, b, rm, rv, False, 0.1, 1e-5)[0].flatten()],
            "outputs": ["float32;n=%d" % total], "grid": ((total + 255) // 256,), "atol": 1e-4}

def run(inputs, kernel):
    return [kernel(*inputs)]
