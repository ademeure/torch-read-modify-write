"""011_model.py:28 [blocks.0] | att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

Inputs (5.0KB total):
  primals_8    float32[1x1x16x16]  1.0KB
  mul          float32[2x2x16x16]  4.0KB
Outputs (4.0KB total):
  masked_fill  float32[2x2x16x16]  4.0KB
Ops: alias, eq, masked_fill  (3 ops)

    kbox iterate fw_011_masked_fill.py
"""
import torch


def init_once():
    return {"h5": "data/fw_011_masked_fill.h5"}


def run(inputs):
    alias = torch.ops.aten.alias.default(inputs.primals_8)
    eq = torch.ops.aten.eq.Scalar(alias, 0)
    masked_fill = torch.ops.aten.masked_fill.Scalar(inputs.mul, eq, -inf)
    return [masked_fill]
