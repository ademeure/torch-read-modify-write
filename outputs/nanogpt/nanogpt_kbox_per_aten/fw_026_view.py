"""026_model.py:30 [blocks.1] | y = att @ v

Inputs (8.0KB total):
  _softmax_1   float32[2x2x16x16]  strides=(512,256,16,1) C  4.0KB
  transpose_7  float32[2x2x16x16]  strides=(1536,16,96,1) NC  4.0KB
Outputs (4.0KB total):
  view_22      float32[2x2x16x16]  strides=(512,256,16,1) C  4.0KB
Ops: expand x2, view x2, clone, _unsafe_view, bmm  (7 ops)

    kbox iterate fw_026_view.py
"""
import torch


def init_once():
    return {"h5": "data/fw_026_view.h5"}


def run(inputs):
    # _softmax_1: strides=(512,256,16,1) C
    # transpose_7: strides=(1536,16,96,1) NC
    expand_6 = torch.ops.aten.expand.default(inputs._softmax_1, [2, 2, 16, 16])  # strides=(512,256,16,1) C
    view_21 = torch.ops.aten.view.default(expand_6, [4, 16, 16])  # strides=(256,16,1) C
    expand_7 = torch.ops.aten.expand.default(inputs.transpose_7, [2, 2, 16, 16])  # strides=(1536,16,96,1) NC
    clone_6 = torch.ops.aten.clone.default(expand_7, memory_format=torch.contiguous_format)  # strides=(512,256,16,1) C
    _unsafe_view_5 = torch.ops.aten._unsafe_view.default(clone_6, [4, 16, 16])  # strides=(256,16,1) C
    bmm_3 = torch.ops.aten.bmm.default(view_21, _unsafe_view_5)  # strides=(256,16,1) C
    view_22 = torch.ops.aten.view.default(bmm_3, [2, 2, 16, 16])  # strides=(512,256,16,1) C
    return [view_22]
