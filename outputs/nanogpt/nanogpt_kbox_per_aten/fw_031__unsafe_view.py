"""031_model.py:72 [lm_head] | logits = self.lm_head(x)

Inputs (12.0KB total):
  primals_2       float32[64x32]  strides=(32,1) C  8.0KB
  getitem_18      float32[2x16x32]  strides=(512,32,1) C  4.0KB
Outputs (8.0KB total):
  _unsafe_view_6  float32[2x16x64]  strides=(1024,64,1) C  8.0KB
Ops: t, view, mm, _unsafe_view  (4 ops)

    kbox iterate fw_031__unsafe_view.py
"""
import torch


def init_once():
    return {"h5": "data/fw_031__unsafe_view.h5"}


def run(inputs):
    # primals_2: strides=(32,1) C
    # getitem_18: strides=(512,32,1) C
    t_8 = torch.ops.aten.t.default(inputs.primals_2)  # strides=(1,32) NC
    view_30 = torch.ops.aten.view.default(inputs.getitem_18, [32, 32])  # strides=(32,1) C
    mm = torch.ops.aten.mm.default(view_30, t_8)  # strides=(64,1) C
    _unsafe_view_6 = torch.ops.aten._unsafe_view.default(mm, [2, 16, 64])  # strides=(1024,64,1) C
    return [_unsafe_view_6]
