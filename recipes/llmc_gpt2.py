"""Recipe for karpathy/llm.c train_gpt2.py — pretrained GPT-2 124M.

Expects the repo at ``repos/llm.c`` (cloned automatically if missing).
Loads the real GPT-2 124M weights from HuggingFace via the repo's
``GPT.from_pretrained('gpt2')`` method, then does a few fine-tuning
steps on random token data (the architecture + weights are the point,
not the dataset).

Use with ``--max-intermediates-mb`` and ``--storage-dtype bfloat16`` to
keep disk usage reasonable for a 124M-param model.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

_REPO_DIR = Path(__file__).resolve().parent.parent / "repos" / "llm.c"


def _ensure_repo():
    if not (_REPO_DIR / "train_gpt2.py").exists():
        print(f"Cloning llm.c into {_REPO_DIR} …")
        subprocess.check_call(
            ["git", "clone", "--depth", "1",
             "https://github.com/karpathy/llm.c.git", str(_REPO_DIR)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )


def setup() -> dict:
    _ensure_repo()

    sys.path.insert(0, str(_REPO_DIR))
    from train_gpt2 import GPT  # type: ignore

    # Load real pretrained GPT-2 124M weights from HuggingFace
    print("Loading pretrained GPT-2 124M from HuggingFace …")
    model = GPT.from_pretrained("gpt2")
    model.to(torch.bfloat16)
    model.train()

    batch_size = 2
    seq_len = 64

    def get_batch(step: int):
        torch.manual_seed(1337 + step)
        x = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))
        y = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))
        return (x, y), {}

    sample_args, _ = get_batch(0)

    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=6e-4,
        betas=(0.9, 0.95),
        device_type="cpu",
        zero_stage=0,
    )

    return {
        "model": model,
        "sample_args": sample_args,
        "loss_fn": lambda out: out[1],
        "get_batch": get_batch,
        "optimizer": optimizer,
    }
