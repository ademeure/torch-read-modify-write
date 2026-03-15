"""Linear (2 instances: blocks_0, blocks_1)
qkv = self.c_attn(x)

Inputs (16.4KB total):
  getitem    float32[2x16x32]  strides=(512,32,1) C  4.0KB
  primals_6  float32[96x32]  strides=(32,1) C  12.0KB
  primals_7  float32[96]  strides=(1) C  384B
Outputs (12.0KB total):
  view_1     float32[2x16x96]  strides=(1536,96,1) C  12.0KB
Ops: view x2, t, addmm  (4 ops)

    kbox iterate grp_linear_model_L22.py
"""
import torch


def init_once():
    return {"h5_suite": "data/grp_linear_model_L22/"}


def run(inputs):
    # getitem: strides=(512,32,1) C
    # primals_6: strides=(32,1) C
    # primals_7: strides=(1) C
    view = torch.ops.aten.view.default(inputs.getitem, [32, 32])  # strides=(32,1) C
    t = torch.ops.aten.t.default(inputs.primals_6)  # strides=(1,32) NC
    addmm = torch.ops.aten.addmm.default(inputs.primals_7, view, t)  # strides=(96,1) C
    view_1 = torch.ops.aten.view.default(addmm, [2, 16, 96])  # strides=(1536,96,1) C
    return [view_1]
