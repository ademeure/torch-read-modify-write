"""018_model.py:22 [blocks.1] | qkv = self.c_attn(x)

Inputs (16.4KB total):
  getitem_9   float32[2x16x32]  strides=(512,32,1) C  4.0KB
  primals_19  float32[96x32]  strides=(32,1) C  12.0KB
  primals_20  float32[96]  strides=(1) C  384B
Outputs (12.0KB total):
  view_16     float32[2x16x96]  strides=(1536,96,1) C  12.0KB
Ops: view x2, t, addmm  (4 ops)

    kbox iterate fw_018_view.py
"""
import torch


def init_once():
    return {"h5": "data/fw_018_view.h5"}


def run(inputs):
    # getitem_9: strides=(512,32,1) C
    # primals_19: strides=(32,1) C
    # primals_20: strides=(1) C
    view_15 = torch.ops.aten.view.default(inputs.getitem_9, [32, 32])  # strides=(32,1) C
    t_4 = torch.ops.aten.t.default(inputs.primals_19)  # strides=(1,32) NC
    addmm_4 = torch.ops.aten.addmm.default(inputs.primals_20, view_15, t_4)  # strides=(96,1) C
    view_16 = torch.ops.aten.view.default(addmm_4, [2, 16, 96])  # strides=(1536,96,1) C
    return [view_16]
