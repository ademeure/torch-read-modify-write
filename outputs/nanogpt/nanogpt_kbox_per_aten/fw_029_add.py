"""029_model.py:49 [blocks.1] | x = x + self.mlp(self.ln_2(x))

Inputs (36.9KB total):
  add_3       float32[2x16x32]  strides=(512,32,1) C  4.0KB
  primals_24  float32[32]  strides=(1) C  128B
  primals_25  float32[32]  strides=(1) C  128B
  primals_26  float32[128x32]  strides=(32,1) C  16.0KB
  primals_27  float32[128]  strides=(1) C  512B
  primals_28  float32[32x128]  strides=(128,1) C  16.0KB
  primals_29  float32[32]  strides=(1) C  128B
Outputs (4.0KB total):
  add_4       float32[2x16x32]  strides=(512,32,1) C  4.0KB
Ops: view x4, t x2, addmm x2, native_layer_norm, gelu, add, getitem x3  (14 ops)

    kbox iterate fw_029_add.py
"""
import torch
import operator


def init_once():
    return {"h5": "data/fw_029_add.h5"}


def run(inputs):
    # add_3: strides=(512,32,1) C
    # primals_24: strides=(1) C
    # primals_25: strides=(1) C
    # primals_26: strides=(32,1) C
    # primals_27: strides=(1) C
    # primals_28: strides=(128,1) C
    # primals_29: strides=(1) C
    native_layer_norm_3 = torch.ops.aten.native_layer_norm.default(inputs.add_3, [32], inputs.primals_24, inputs.primals_25, 1e-05)  # strides=(512,32,1) C
    getitem_15 = operator.getitem(native_layer_norm_3, 0)  # strides=(512,32,1) C
    getitem_16 = operator.getitem(native_layer_norm_3, 1)  # strides=(16,1,1) C
    getitem_17 = operator.getitem(native_layer_norm_3, 2)  # strides=(16,1,1) C
    view_26 = torch.ops.aten.view.default(getitem_15, [32, 32])  # strides=(32,1) C
    t_6 = torch.ops.aten.t.default(inputs.primals_26)  # strides=(1,32) NC
    addmm_6 = torch.ops.aten.addmm.default(inputs.primals_27, view_26, t_6)  # strides=(128,1) C
    view_27 = torch.ops.aten.view.default(addmm_6, [2, 16, 128])  # strides=(2048,128,1) C
    gelu_1 = torch.ops.aten.gelu.default(view_27)  # strides=(2048,128,1) C
    view_28 = torch.ops.aten.view.default(gelu_1, [32, 128])  # strides=(128,1) C
    t_7 = torch.ops.aten.t.default(inputs.primals_28)  # strides=(1,128) NC
    addmm_7 = torch.ops.aten.addmm.default(inputs.primals_29, view_28, t_7)  # strides=(32,1) C
    view_29 = torch.ops.aten.view.default(addmm_7, [2, 16, 32])  # strides=(512,32,1) C
    add_4 = torch.ops.aten.add.Tensor(inputs.add_3, view_29)  # strides=(512,32,1) C
    return [add_4]
