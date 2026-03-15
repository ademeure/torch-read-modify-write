"""030_model.py:71 [ln_f] | x = self.ln_f(x)

Inputs (4.2KB total):
  add_4       float32[2x16x32]  4.0KB
  primals_30  float32[32]  128B
  primals_31  float32[32]  128B
Outputs (4.2KB total):
  getitem_18  float32[2x16x32]  4.0KB
  getitem_19  float32[2x16x1]  128B
  getitem_20  float32[2x16x1]  128B
Ops: native_layer_norm, getitem x3  (4 ops)

    kbox iterate fw_030_native_layer_norm.py
"""
import torch
import operator


def init_once():
    return {"h5": "data/fw_030_native_layer_norm.h5"}


def run(inputs):
    native_layer_norm_4 = torch.ops.aten.native_layer_norm.default(inputs.add_4, [32], inputs.primals_30, inputs.primals_31, 1e-05)
    getitem_18 = operator.getitem(native_layer_norm_4, 0)
    getitem_19 = operator.getitem(native_layer_norm_4, 1)
    getitem_20 = operator.getitem(native_layer_norm_4, 2)
    return [getitem_18, getitem_19, getitem_20]
