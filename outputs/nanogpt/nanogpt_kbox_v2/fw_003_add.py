"""003_model.py:68 | x = self.wte(idx) + self.wpe(pos)

Inputs (6.0KB total):
  embedding    float32[2x16x32]  4.0KB
  embedding_1  float32[16x32]  2.0KB
Outputs (4.0KB total):
  add          float32[2x16x32]  4.0KB
Ops: add  (1 ops)

    kbox iterate fw_003_add.py
"""
import torch


def init_once():
    return {"h5": "data/fw_003_add.h5"}


def run(inputs):
    add = torch.ops.aten.add.Tensor(inputs.embedding, inputs.embedding_1)
    return [add]
