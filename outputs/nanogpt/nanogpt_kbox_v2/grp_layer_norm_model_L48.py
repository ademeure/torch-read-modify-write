"""LayerNorm (2 instances: blocks_0, blocks_1)
x = x + self.attn(self.ln_1(x))

Inputs (8.2KB total):
  add        float32[2x16x32]  4.0KB
  primals_4  float32[32]  128B
  primals_5  float32[32]  128B
  view_10    float32[2x16x32]  4.0KB
Outputs (4.0KB total):
  add_1      float32[2x16x32]  4.0KB
Ops: native_layer_norm, add, getitem x3  (5 ops)

    kbox iterate grp_layer_norm_model_L48.py
"""
import torch
import operator


def init_once():
    return {"h5_suite": "data/grp_layer_norm_model_L48/"}


def run(inputs):
    native_layer_norm = torch.ops.aten.native_layer_norm.default(inputs.add, [32], inputs.primals_4, inputs.primals_5, 1e-05)
    getitem = operator.getitem(native_layer_norm, 0)
    getitem_1 = operator.getitem(native_layer_norm, 1)
    getitem_2 = operator.getitem(native_layer_norm, 2)
    add_1 = torch.ops.aten.add.Tensor(inputs.add, inputs.view_10)
    return [add_1]
