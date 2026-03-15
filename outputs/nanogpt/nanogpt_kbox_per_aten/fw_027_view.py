"""027_model.py:31 [blocks.1] | y = y.transpose(1, 2).contiguous().view(B, T, C)

Inputs (4.0KB total):
  view_22  float32[2x2x16x16]  strides=(512,256,16,1) C  4.0KB
Outputs (4.0KB total):
  view_23  float32[2x16x32]  strides=(512,32,1) C  4.0KB
Ops: transpose, clone, view  (3 ops)

    kbox iterate fw_027_view.py
"""
import torch


def init_once():
    return {"h5": "data/fw_027_view.h5"}


def run(inputs):
    # view_22: strides=(512,256,16,1) C
    transpose_9 = torch.ops.aten.transpose.int(inputs.view_22, 1, 2)  # strides=(512,16,256,1) NC
    clone_7 = torch.ops.aten.clone.default(transpose_9, memory_format=torch.contiguous_format)  # strides=(512,32,16,1) C
    view_23 = torch.ops.aten.view.default(clone_7, [2, 16, 32])  # strides=(512,32,1) C
    return [view_23]
