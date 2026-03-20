"""Reference for aten.detach — detach from autograd (no-op copy for reference)."""
import torch
from torch_graph.cuda_ref_kernels._common import check

aten = torch.ops.aten

KERNEL_SRC = ""  # detach is a no-op — no CUDA kernel needed

def test():
    x = torch.randn(1024, device="cuda", requires_grad=True)
    result = aten.detach.default(x)
    expected = x.detach()
    check("aten.detach", result, expected)
    print("PASS aten.detach")

if __name__ == "__main__":
    test()
