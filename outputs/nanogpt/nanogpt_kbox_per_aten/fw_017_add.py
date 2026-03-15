"""017_model.py:48 [blocks.1] | x = x + self.attn(self.ln_1(x))

Inputs (8.2KB total):
  add_2       float32[2x16x32]  4.0KB
  primals_17  float32[32]  128B
  primals_18  float32[32]  128B
  view_25     float32[2x16x32]  4.0KB
Outputs (4.0KB total):
  add_3       float32[2x16x32]  4.0KB
Ops: native_layer_norm, add, getitem x3  (5 ops)

    kbox iterate fw_017_add.py
"""
import torch
import operator


def init_once():
    return {"h5": "data/fw_017_add.h5"}


def run(inputs):
    native_layer_norm_2 = torch.ops.aten.native_layer_norm.default(inputs.add_2, [32], inputs.primals_17, inputs.primals_18, 1e-05)
    getitem_9 = operator.getitem(native_layer_norm_2, 0)
    getitem_10 = operator.getitem(native_layer_norm_2, 1)
    getitem_11 = operator.getitem(native_layer_norm_2, 2)
    add_3 = torch.ops.aten.add.Tensor(inputs.add_2, inputs.view_25)
    return [add_3]
