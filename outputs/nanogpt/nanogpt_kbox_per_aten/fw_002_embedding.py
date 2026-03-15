"""002_model.py:68 [wpe] | x = self.wte(idx) + self.wpe(pos)

Inputs (2.1KB total):
  primals_3    float32[16x32]  strides=(32,1) C  2.0KB
  arange       int64[16]  strides=(1) C  128B
Outputs (2.0KB total):
  embedding_1  float32[16x32]  strides=(32,1) C  2.0KB
Ops: embedding  (1 ops)

    kbox iterate fw_002_embedding.py
"""
import torch


def init_once():
    return {"h5": "data/fw_002_embedding.h5"}


def run(inputs):
    # primals_3: strides=(32,1) C
    # arange: strides=(1) C
    embedding_1 = torch.ops.aten.embedding.default(inputs.primals_3, inputs.arange)  # strides=(32,1) C
    return [embedding_1]
