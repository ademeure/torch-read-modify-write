"""006_model.py:23 [blocks.0] | q, k, v = qkv.split(C, dim=2)

Inputs (12.0KB total):
  view_1     float32[2x16x96]  12.0KB
Outputs (12.0KB total):
  getitem_3  float32[2x16x32]  4.0KB
  getitem_4  float32[2x16x32]  4.0KB
  getitem_5  float32[2x16x32]  4.0KB
Ops: split, getitem x3  (4 ops)

    kbox iterate fw_006_split.py
"""
import torch
import operator


def init_once():
    return {"h5": "data/fw_006_split.h5"}


def run(inputs):
    split = torch.ops.aten.split.Tensor(inputs.view_1, 32, 2)
    getitem_3 = operator.getitem(split, 0)
    getitem_4 = operator.getitem(split, 1)
    getitem_5 = operator.getitem(split, 2)
    return [getitem_3, getitem_4, getitem_5]
