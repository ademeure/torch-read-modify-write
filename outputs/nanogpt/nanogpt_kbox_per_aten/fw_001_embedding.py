"""001_model.py:68 [wte] | x = self.wte(idx) + self.wpe(pos)

Inputs (8.2KB total):
  primals_2  float32[64x32]  strides=(32,1) C  8.0KB
  primals_1  int64[2x16]  strides=(16,1) C  256B
Outputs (4.0KB total):
  embedding  float32[2x16x32]  strides=(512,32,1) C  4.0KB
Ops: embedding  (1 ops)

    kbox iterate fw_001_embedding.py
"""
import torch


def init_once():
    return {"h5": "data/fw_001_embedding.h5"}


def run(inputs):
    # primals_2: strides=(32,1) C
    # primals_1: strides=(16,1) C
    embedding = torch.ops.aten.embedding.default(inputs.primals_2, inputs.primals_1)  # strides=(512,32,1) C
    return [embedding]
