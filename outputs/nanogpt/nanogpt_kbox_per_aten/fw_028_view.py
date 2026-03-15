"""028_model.py:32 [blocks.1] | return self.c_proj(y)

Inputs (8.1KB total):
  view_23     float32[2x16x32]  4.0KB
  primals_22  float32[32x32]  4.0KB
  primals_23  float32[32]  128B
Outputs (4.0KB total):
  view_25     float32[2x16x32]  4.0KB
Ops: view x2, t, addmm  (4 ops)

    kbox iterate fw_028_view.py
"""
import torch


def init_once():
    return {"h5": "data/fw_028_view.h5"}


def run(inputs):
    view_24 = torch.ops.aten.view.default(inputs.view_23, [32, 32])
    t_5 = torch.ops.aten.t.default(inputs.primals_22)
    addmm_5 = torch.ops.aten.addmm.default(inputs.primals_23, view_24, t_5)
    view_25 = torch.ops.aten.view.default(addmm_5, [2, 16, 32])
    return [view_25]
