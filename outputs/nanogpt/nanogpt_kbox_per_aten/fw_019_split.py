"""019_model.py:23 [blocks.1] | q, k, v = qkv.split(C, dim=2)

Inputs (12.0KB total):
  view_16     float32[2x16x96]  strides=(1536,96,1) C  12.0KB
Outputs (12.0KB total):
  getitem_12  float32[2x16x32]  strides=(1536,96,1) NC  4.0KB
  getitem_13  float32[2x16x32]  strides=(1536,96,1) NC  4.0KB
  getitem_14  float32[2x16x32]  strides=(1536,96,1) NC  4.0KB
Ops: split, getitem x3  (4 ops)

    kbox iterate fw_019_split.py
"""
import torch
import operator


def init_once():
    return {"h5": "data/fw_019_split.h5"}


def run(inputs):
    # view_16: strides=(1536,96,1) C
    split_1 = torch.ops.aten.split.Tensor(inputs.view_16, 32, 2)  # strides=(1536,96,1) NC
    getitem_12 = operator.getitem(split_1, 0)  # strides=(1536,96,1) NC
    getitem_13 = operator.getitem(split_1, 1)  # strides=(1536,96,1) NC
    getitem_14 = operator.getitem(split_1, 2)  # strides=(1536,96,1) NC
    return [getitem_12, getitem_13, getitem_14]
