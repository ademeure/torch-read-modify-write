"""LayerNorm (2 instances: blocks_0, blocks_1)
x = x + self.mlp(self.ln_2(x))

Inputs (36.9KB total):
  add_1       float32[2x16x32]  strides=(512,32,1) C  4.0KB
  primals_11  float32[32]  strides=(1) C  128B
  primals_12  float32[32]  strides=(1) C  128B
  primals_13  float32[128x32]  strides=(32,1) C  16.0KB
  primals_14  float32[128]  strides=(1) C  512B
  primals_15  float32[32x128]  strides=(128,1) C  16.0KB
  primals_16  float32[32]  strides=(1) C  128B
Outputs (4.0KB total):
  add_2       float32[2x16x32]  strides=(512,32,1) C  4.0KB
Ops: view x4, t x2, addmm x2, native_layer_norm, gelu, add, getitem x3  (14 ops)

    kbox iterate grp_layer_norm_model_L49.py
"""
import torch
import operator


def init_once():
    return {"h5_suite": "data/grp_layer_norm_model_L49/"}


def run(inputs):
    # add_1: strides=(512,32,1) C
    # primals_11: strides=(1) C
    # primals_12: strides=(1) C
    # primals_13: strides=(32,1) C
    # primals_14: strides=(1) C
    # primals_15: strides=(128,1) C
    # primals_16: strides=(1) C
    native_layer_norm_1 = torch.ops.aten.native_layer_norm.default(inputs.add_1, [32], inputs.primals_11, inputs.primals_12, 1e-05)  # strides=(512,32,1) C
    getitem_6 = operator.getitem(native_layer_norm_1, 0)  # strides=(512,32,1) C
    getitem_7 = operator.getitem(native_layer_norm_1, 1)  # strides=(16,1,1) C
    getitem_8 = operator.getitem(native_layer_norm_1, 2)  # strides=(16,1,1) C
    view_11 = torch.ops.aten.view.default(getitem_6, [32, 32])  # strides=(32,1) C
    t_2 = torch.ops.aten.t.default(inputs.primals_13)  # strides=(1,32) NC
    addmm_2 = torch.ops.aten.addmm.default(inputs.primals_14, view_11, t_2)  # strides=(128,1) C
    view_12 = torch.ops.aten.view.default(addmm_2, [2, 16, 128])  # strides=(2048,128,1) C
    gelu = torch.ops.aten.gelu.default(view_12)  # strides=(2048,128,1) C
    view_13 = torch.ops.aten.view.default(gelu, [32, 128])  # strides=(128,1) C
    t_3 = torch.ops.aten.t.default(inputs.primals_15)  # strides=(1,128) NC
    addmm_3 = torch.ops.aten.addmm.default(inputs.primals_16, view_13, t_3)  # strides=(32,1) C
    view_14 = torch.ops.aten.view.default(addmm_3, [2, 16, 32])  # strides=(512,32,1) C
    add_2 = torch.ops.aten.add.Tensor(inputs.add_1, view_14)  # strides=(512,32,1) C
    return [add_2]
