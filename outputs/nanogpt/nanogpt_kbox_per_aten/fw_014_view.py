"""014_model.py:31 [blocks.0] | y = y.transpose(1, 2).contiguous().view(B, T, C)

Inputs (4.0KB total):
  view_7  float32[2x2x16x16]  strides=(512,256,16,1) C  4.0KB
Outputs (4.0KB total):
  view_8  float32[2x16x32]  strides=(512,32,1) C  4.0KB
Ops: transpose, clone, view  (3 ops)

    kbox iterate fw_014_view.py
"""
import torch


def init_once():
    return {"h5": "data/fw_014_view.h5"}


def run(inputs):
    # view_7: strides=(512,256,16,1) C
    transpose_4 = torch.ops.aten.transpose.int(inputs.view_7, 1, 2)  # strides=(512,16,256,1) NC
    clone_3 = torch.ops.aten.clone.default(transpose_4, memory_format=torch.contiguous_format)  # strides=(512,32,16,1) C
    view_8 = torch.ops.aten.view.default(clone_3, [2, 16, 32])  # strides=(512,32,1) C
    return [view_8]
