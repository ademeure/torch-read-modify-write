"""Reference for aten.alias — identity op, returns same data (no-op copy for reference)."""
import torch
from torch_graph.cuda_ref_kernels._common import check

aten = torch.ops.aten

KERNEL_SRC = ""  # alias is a no-op — no CUDA kernel needed

def test():
    x = torch.randn(1024, device="cuda")
    result = aten.alias.default(x)
    check("aten.alias", result, x)
    print("PASS aten.alias")

if __name__ == "__main__":
    test()
