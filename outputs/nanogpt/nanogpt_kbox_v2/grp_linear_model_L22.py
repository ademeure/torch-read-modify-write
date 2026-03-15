"""Linear (2 instances: blocks_0, blocks_1)
qkv = self.c_attn(x)

Inputs (16.4KB total):
  getitem    float32[2x16x32]  4.0KB
  primals_6  float32[96x32]  12.0KB
  primals_7  float32[96]  384B
Outputs (12.0KB total):
  view_1     float32[2x16x96]  12.0KB
Ops: view x2, t, addmm  (4 ops)

    kbox iterate grp_linear_model_L22.py
"""
import torch


def init_once():
    return {"h5_suite": "data/grp_linear_model_L22/"}


def run(inputs):
    view = torch.ops.aten.view.default(inputs.getitem, [32, 32])
    t = torch.ops.aten.t.default(inputs.primals_6)
    addmm = torch.ops.aten.addmm.default(inputs.primals_7, view, t)
    view_1 = torch.ops.aten.view.default(addmm, [2, 16, 96])
    return [view_1]
