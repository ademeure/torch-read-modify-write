"""012_model.py:29 [blocks.0] | att = F.softmax(att, dim=-1)

Inputs (4.0KB total):
  masked_fill  float32[2x2x16x16]  strides=(512,256,16,1) C  4.0KB
Outputs (4.0KB total):
  detach       float32[2x2x16x16]  strides=(512,256,16,1) C  4.0KB
Ops: _softmax, detach  (2 ops)

    kbox iterate fw_012_detach.py
"""
import torch


def init_once():
    return {"h5": "data/fw_012_detach.h5"}


def run(inputs):
    # masked_fill: strides=(512,256,16,1) C
    _softmax = torch.ops.aten._softmax.default(inputs.masked_fill, -1, False)  # strides=(512,256,16,1) C
    detach = torch.ops.aten.detach.default(_softmax)  # strides=(512,256,16,1) C
    return [detach]
