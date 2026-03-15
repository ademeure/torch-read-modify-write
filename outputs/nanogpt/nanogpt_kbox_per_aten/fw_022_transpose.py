"""022_model.py:26 [blocks.1] | v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

Inputs (4.0KB total):
  getitem_14   float32[2x16x32]  strides=(1536,96,1) NC  4.0KB
Outputs (4.0KB total):
  transpose_7  float32[2x2x16x16]  strides=(1536,16,96,1) NC  4.0KB
Ops: view, transpose  (2 ops)

    kbox iterate fw_022_transpose.py
"""
import torch


def init_once():
    return {"h5": "data/fw_022_transpose.h5"}


def run(inputs):
    # getitem_14: strides=(1536,96,1) NC
    view_19 = torch.ops.aten.view.default(inputs.getitem_14, [2, 16, 2, 16])  # strides=(1536,96,16,1) NC
    transpose_7 = torch.ops.aten.transpose.int(view_19, 1, 2)  # strides=(1536,16,96,1) NC
    return [transpose_7]
