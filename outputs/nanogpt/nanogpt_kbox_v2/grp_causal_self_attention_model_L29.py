"""CausalSelfAttention (2 instances: blocks_0, blocks_1)
att = F.softmax(att, dim=-1)

Inputs (4.0KB total):
  masked_fill  float32[2x2x16x16]  4.0KB
Outputs (4.0KB total):
  detach       float32[2x2x16x16]  4.0KB
Ops: _softmax, detach  (2 ops)

    kbox iterate grp_causal_self_attention_model_L29.py
"""
import torch


def init_once():
    return {"h5_suite": "data/grp_causal_self_attention_model_L29/"}


def run(inputs):
    _softmax = torch.ops.aten._softmax.default(inputs.masked_fill, -1, False)
    detach = torch.ops.aten.detach.default(_softmax)
    return [detach]
