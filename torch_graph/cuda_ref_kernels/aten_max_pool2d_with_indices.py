"""Reference CUDA kernel for aten.max_pool2d_with_indices."""
import torch, numpy as np
from torch_graph.cuda_ref_kernels.aten_max_pool2d import KERNEL_SRC
NN, CC, HH, WW, KH, KW, SH, SW, PH, PW = 1, 4, 8, 8, 2, 2, 2, 2, 0, 0
OH, OW = (HH + 2*PH - KH) // SH + 1, (WW + 2*PW - KW) // SW + 1
def init_once():
    x = torch.randn(NN, CC, HH, WW, device="cuda")
    total = NN * CC * OH * OW
    return {"kernel_source": KERNEL_SRC, "inputs": [x.contiguous()],
            "expected": [torch.ops.aten.max_pool2d_with_indices.default(x, [KH,KW], [SH,SW])[0].flatten()],
            "outputs": ["float32;n=%d" % total], "grid": ((total + 255) // 256,)}
def run(inputs, kernel):
    total = NN * CC * OH * OW
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(NN), np.uint32(CC), np.uint32(HH), np.uint32(WW),
        np.uint32(KH), np.uint32(KW), np.uint32(SH), np.uint32(SW),
        np.uint32(PH), np.uint32(PW), np.uint32(OH), np.uint32(OW)])]
