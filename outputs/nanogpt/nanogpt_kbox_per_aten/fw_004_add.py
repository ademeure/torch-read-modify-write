"""004_model.py:48 [blocks.0] | x = x + self.attn(self.ln_1(x))

Inputs (8.2KB total):
  add        float32[2x16x32]  strides=(512,32,1) C  4.0KB
  primals_4  float32[32]  strides=(1) C  128B
  primals_5  float32[32]  strides=(1) C  128B
  view_10    float32[2x16x32]  strides=(512,32,1) C  4.0KB
Outputs (4.0KB total):
  add_1      float32[2x16x32]  strides=(512,32,1) C  4.0KB
Ops: native_layer_norm, add, getitem x3  (5 ops)

    kbox iterate fw_004_add.py
"""
import torch
import operator


def init_once():
    return {"h5": "data/fw_004_add.h5"}


def run(inputs):
    # add: strides=(512,32,1) C
    # primals_4: strides=(1) C
    # primals_5: strides=(1) C
    # view_10: strides=(512,32,1) C
    native_layer_norm = torch.ops.aten.native_layer_norm.default(inputs.add, [32], inputs.primals_4, inputs.primals_5, 1e-05)  # strides=(512,32,1) C
    getitem = operator.getitem(native_layer_norm, 0)  # strides=(512,32,1) C
    getitem_1 = operator.getitem(native_layer_norm, 1)  # strides=(16,1,1) C
    getitem_2 = operator.getitem(native_layer_norm, 2)  # strides=(16,1,1) C
    add_1 = torch.ops.aten.add.Tensor(inputs.add, inputs.view_10)  # strides=(512,32,1) C
    return [add_1]
