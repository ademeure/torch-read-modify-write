"""025_model.py:29 [blocks.1] | att = F.softmax(att, dim=-1)

Inputs (4.0KB total):
  masked_fill_1  float32[2x2x16x16]  strides=(512,256,16,1) C  4.0KB
Outputs (4.0KB total):
  detach_1       float32[2x2x16x16]  strides=(512,256,16,1) C  4.0KB
Ops: _softmax, detach  (2 ops)

    kbox iterate fw_025_detach.py
"""
import torch


def init_once():
    return {"h5": "data/fw_025_detach.h5"}


def run(inputs):
    # masked_fill_1: strides=(512,256,16,1) C
    _softmax_1 = torch.ops.aten._softmax.default(inputs.masked_fill_1, -1, False)  # strides=(512,256,16,1) C
    detach_1 = torch.ops.aten.detach.default(_softmax_1)  # strides=(512,256,16,1) C
    return [detach_1]
