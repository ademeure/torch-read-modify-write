"""024_model.py:28 [blocks.1] | att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

Inputs (5.0KB total):
  primals_21     float32[1x1x16x16]  1.0KB
  mul_1          float32[2x2x16x16]  4.0KB
Outputs (4.0KB total):
  masked_fill_1  float32[2x2x16x16]  4.0KB
Ops: alias, eq, masked_fill  (3 ops)

    kbox iterate fw_024_masked_fill.py
"""
import torch


def init_once():
    return {"h5": "data/fw_024_masked_fill.h5"}


def run(inputs):
    alias_1 = torch.ops.aten.alias.default(inputs.primals_21)
    eq_1 = torch.ops.aten.eq.Scalar(alias_1, 0)
    masked_fill_1 = torch.ops.aten.masked_fill.Scalar(inputs.mul_1, eq_1, -inf)
    return [masked_fill_1]
