"""Reference CUDA kernel for aten.max_pool2d_with_indices."""
import torch
from torch_graph.cuda_ref_kernels.aten_max_pool2d import KERNEL_SRC

def init_once():
    x = torch.randn(1, 4, 8, 8, device="cuda")
    expected = torch.ops.aten.max_pool2d_with_indices.default(x, [2,2], [2,2])
    total = expected[0].numel()
    return {"kernel_source": KERNEL_SRC, "inputs": [x.contiguous()],
            "expected": [expected[0].flatten()],
            "outputs": ["float32;n=%d" % total],
            "grid": ((total + 255) // 256,), "block": (256,)}

def run(inputs, kernel):
    return [kernel(*inputs)]
