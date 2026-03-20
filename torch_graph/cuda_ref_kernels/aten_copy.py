"""Reference CUDA kernel for aten.copy — copy tensor data."""
import torch
from torch_graph.cuda_ref_kernels.aten_clone import KERNEL_SRC

def init_once():
    x = torch.randn(1024, device="cuda")
    return {"kernel_source": KERNEL_SRC, "inputs": [x], "expected": [torch.ops.aten.clone.default(x)]}

def run(inputs, kernel):
    return [kernel(*inputs)]
