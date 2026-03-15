"""Template recipe — copy this and fill in for your model.

A recipe is a Python file with a ``setup()`` function that returns a dict
describing your model, its inputs, and how to compute the loss.

Usage:
    python extract_repo.py --recipe recipes/my_model.py --warmup 10
"""

import torch
import torch.nn as nn


def setup() -> dict:
    # ── 1. Build or load your model ──────────────────────────────
    # Option A: import from a cloned repo
    #   import sys; sys.path.insert(0, "repos/my_repo")
    #   from my_repo.model import MyModel
    #   model = MyModel(...)
    #
    # Option B: load a checkpoint
    #   model = MyModel(...)
    #   model.load_state_dict(torch.load("checkpoint.pt")["model"])
    #
    # Option C: inline definition (for quick tests)
    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )
    model.train()

    # ── 2. Provide one real batch ────────────────────────────────
    # These are the positional args to model.forward().
    x = torch.randn(8, 128)
    sample_args = (x,)

    # ── 3. Loss function ─────────────────────────────────────────
    # Required when the model does NOT return a scalar loss directly.
    #
    # If your model returns a single loss tensor: loss_fn = None
    # If it returns (logits, loss):  loss_fn = lambda out: out[1]
    # If it returns logits only:     loss_fn = lambda out: out.sum()
    targets = torch.randint(0, 10, (8,))
    loss_fn = lambda out: nn.functional.cross_entropy(out, targets)

    # ── 4. (Optional) batch generator for warmup steps ───────────
    # If you want the warmup steps to use fresh data each iteration,
    # provide a callable: get_batch(step_number) -> (args_tuple, kwargs_dict)
    #
    # def get_batch(step):
    #     x = load_batch_from_dataset(step)
    #     return (x,), {}
    get_batch = None

    # ── 5. (Optional) optimizer ──────────────────────────────────
    # If None, AdamW with lr=1e-4 is used.
    optimizer = None

    return {
        "model": model,
        "sample_args": sample_args,
        "loss_fn": loss_fn,
        "get_batch": get_batch,
        "optimizer": optimizer,
    }
