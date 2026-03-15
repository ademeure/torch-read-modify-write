"""CausalSelfAttention (2 instances: blocks_0, blocks_1)
y = att @ v

Inputs (8.0KB total):
  _softmax     float32[2x2x16x16]  4.0KB
  transpose_2  float32[2x2x16x16]  4.0KB
Outputs (4.0KB total):
  view_7       float32[2x2x16x16]  4.0KB
Ops: expand x2, view x2, clone, _unsafe_view, bmm  (7 ops)

    kbox iterate grp_causal_self_attention_model_L30.py
"""
import torch


def init_once():
    return {"h5_suite": "data/grp_causal_self_attention_model_L30/"}


def run(inputs):
    expand_2 = torch.ops.aten.expand.default(inputs._softmax, [2, 2, 16, 16])
    view_6 = torch.ops.aten.view.default(expand_2, [4, 16, 16])
    expand_3 = torch.ops.aten.expand.default(inputs.transpose_2, [2, 2, 16, 16])
    clone_2 = torch.ops.aten.clone.default(expand_3, memory_format=torch.contiguous_format)
    _unsafe_view_2 = torch.ops.aten._unsafe_view.default(clone_2, [4, 16, 16])
    bmm_1 = torch.ops.aten.bmm.default(view_6, _unsafe_view_2)
    view_7 = torch.ops.aten.view.default(bmm_1, [2, 2, 16, 16])
    return [view_7]
