"""Recipe for karpathy/nanoGPT with real Shakespeare character-level data.

Expects the repo at ``repos/nanoGPT`` (the CLI will clone it if missing).
Runs ``data/shakespeare_char/prepare.py`` automatically if train.bin is absent.

Model size matches the official ``train_shakespeare_char.py`` config
(6 layers, 6 heads, 384-dim) — this is the real architecture, not a toy.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_DIR = Path(__file__).resolve().parent.parent / "repos" / "nanoGPT"


def _ensure_repo():
    if not (_REPO_DIR / "model.py").exists():
        print(f"Cloning nanoGPT into {_REPO_DIR} …")
        subprocess.check_call(
            ["git", "clone", "--depth", "1",
             "https://github.com/karpathy/nanoGPT.git", str(_REPO_DIR)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )


def _ensure_data():
    data_dir = _REPO_DIR / "data" / "shakespeare_char"
    if not (data_dir / "train.bin").exists():
        print("Preparing shakespeare_char dataset …")
        subprocess.check_call(
            [sys.executable, str(data_dir / "prepare.py")],
            cwd=str(data_dir),
        )


def setup() -> dict:
    _ensure_repo()
    _ensure_data()

    sys.path.insert(0, str(_REPO_DIR))
    from model import GPT, GPTConfig  # type: ignore

    # ── model ─────────────────────────────────────────────────────
    # Official shakespeare_char config (real architecture, CPU-friendly batch)
    config = GPTConfig(
        block_size=256,
        vocab_size=65,
        n_layer=6,
        n_head=6,
        n_embd=384,
        dropout=0.0,
        bias=False,
    )
    model = GPT(config)
    model.train()

    # ── real data ─────────────────────────────────────────────────
    data_path = _REPO_DIR / "data" / "shakespeare_char" / "train.bin"
    data = np.memmap(str(data_path), dtype=np.uint16, mode="r")

    batch_size = 4
    block_size = config.block_size

    def get_batch(step: int):
        torch.manual_seed(1337 + step)
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack(
            [torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix]
        )
        y = torch.stack(
            [torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix]
        )
        return (x, y), {}

    sample_args, sample_kw = get_batch(0)

    # ── optimizer (same param-group logic as the real training script) ─
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=1e-3,
        betas=(0.9, 0.95),
        device_type="cpu",
    )

    return {
        "model": model,
        "sample_args": sample_args,
        "loss_fn": lambda out: out[1],   # GPT.forward returns (logits, loss)
        "get_batch": get_batch,
        "optimizer": optimizer,
    }
