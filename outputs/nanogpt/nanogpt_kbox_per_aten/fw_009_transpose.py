"""009_model.py:26 [blocks.0] | v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

Inputs (4.0KB total):
  getitem_5    float32[2x16x32]  strides=(1536,96,1) NC  4.0KB
Outputs (4.0KB total):
  transpose_2  float32[2x2x16x16]  strides=(1536,16,96,1) NC  4.0KB
Ops: view, transpose  (2 ops)

    kbox iterate fw_009_transpose.py
"""
import torch


def init_once():
    return {"h5": "data/fw_009_transpose.h5"}


def run(inputs):
    # getitem_5: strides=(1536,96,1) NC
    view_4 = torch.ops.aten.view.default(inputs.getitem_5, [2, 16, 2, 16])  # strides=(1536,96,16,1) NC
    transpose_2 = torch.ops.aten.transpose.int(view_4, 1, 2)  # strides=(1536,16,96,1) NC
    return [transpose_2]
