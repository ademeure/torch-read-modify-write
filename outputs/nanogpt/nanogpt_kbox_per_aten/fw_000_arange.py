"""000_model.py:67 | pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

Outputs (128B total):
  arange      int64[16]  128B
Ops: arange  (1 ops)

    kbox iterate fw_000_arange.py
"""
import torch


def init_once():
    return {"h5": "data/fw_000_arange.h5"}


def run(inputs):
    arange = torch.ops.aten.arange.start(0, 16, dtype=torch.int64, device=torch.device("cuda:0"), pin_memory=False)
    return [arange]
