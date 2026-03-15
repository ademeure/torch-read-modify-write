#!/usr/bin/env python3
"""Capture autoresearch aten graphs via auto_install and validate modifications.

Runs the autoresearch GPT model through our auto_install system:
  1. Capture aten forward+backward on step 0
  2. Run N steps through the captured aten graph
  3. Optionally verify that edits to the .py file take effect

Usage:
  .venv/bin/python scripts/autoresearch_capture.py [--steps 10] [--depth 8] [--batch-size 32]
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import types
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

# autoresearch repo
_ar_root = _root / ".autoresearch_repo"
sys.path.insert(0, str(_ar_root))

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault(
    "LIBRARY_PATH",
    "/usr/local/cuda-13.1/targets/x86_64-linux/lib/stubs:"
    + os.environ.get("LIBRARY_PATH", ""),
)

# Force SDPA on non-Hopper GPUs
_fa_spec = importlib.util.spec_from_file_location(
    "flash_attention", str(_ar_root / "flash_attention.py"))
_fa_mod = importlib.util.module_from_spec(_fa_spec)
import builtins
_saved_import = builtins.__import__
def _block_kernels(name, *args, **kwargs):
    if name == "kernels":
        raise ImportError("blocked for SDPA fallback")
    return _saved_import(name, *args, **kwargs)
builtins.__import__ = _block_kernels
try:
    _fa_spec.loader.exec_module(_fa_mod)
finally:
    builtins.__import__ = _saved_import
sys.modules["flash_attention"] = _fa_mod
print(f"Attention backend: {_fa_mod.FLASH_ATTENTION_IMPL}")

import torch
import torch._dynamo
torch.set_float32_matmul_precision("high")


def _import_classes():
    """Import GPT/GPTConfig/MuonAdamW without running the training loop."""
    importlib.import_module("prepare")
    train_path = _ar_root / "train.py"
    source = train_path.read_text()
    marker = "# ---------------------------------------------------------------------------\n# Hyperparameters"
    idx = source.find(marker)
    class_source = source[:idx]
    train_mod = types.ModuleType("train")
    train_mod.__file__ = str(train_path)
    sys.modules["train"] = train_mod
    exec(compile(class_source, str(train_path), "exec"), train_mod.__dict__)
    return train_mod


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--cache-dir", default=".autoresearch_cache")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_mod = _import_classes()
    GPT, GPTConfig, MuonAdamW = train_mod.GPT, train_mod.GPTConfig, train_mod.MuonAdamW
    from prepare import Tokenizer

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    tokenizer = Tokenizer.from_directory()
    vocab_size = tokenizer.get_vocab_size()

    ASPECT_RATIO = 64
    HEAD_DIM = 128
    base_dim = args.depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM

    config = GPTConfig(
        sequence_len=args.seq_len,
        vocab_size=vocab_size,
        n_layer=args.depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern="SSSL",
    )

    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device="cuda")
    model.init_weights()
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.depth}L-{model_dim}d  ({n_params:,} params)")

    optimizer = model.setup_optimizer()
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    # Configure auto_install
    import torch_graph.auto_install as ai
    ai.unpatch()
    ai.configure(
        cache_dir=args.cache_dir,
        force_recapture=True,
        verbose=True,
        capture_backward=True,
        dynamic=False,
        capture_optimizer=True,
        generate_graph=True,   # HTML visualization with Triton kernels
    )
    ai.patch()

    # torch.compile goes through our auto_install
    compiled = torch.compile(model, dynamic=False)

    print(f"\nRunning {args.steps} steps (capture on step 0)...\n")
    losses = []
    for step in range(args.steps):
        torch.manual_seed(1337 + step)
        x = torch.randint(0, vocab_size, (args.batch_size, args.seq_len), device="cuda")
        y = torch.randint(0, vocab_size, (args.batch_size, args.seq_len), device="cuda")

        with autocast_ctx:
            loss = compiled(x, y)
        loss.backward()
        optimizer.step()
        model.zero_grad(set_to_none=True)

        loss_val = loss.item()
        losses.append(loss_val)
        print(f"  step {step:3d}: loss={loss_val:.6f}")

    ai.unpatch()

    # List generated files
    cache = Path(args.cache_dir)
    if cache.exists():
        print(f"\nGenerated files in {args.cache_dir}/:")
        for f in sorted(cache.rglob("*")):
            if f.is_file():
                size = f.stat().st_size
                print(f"  {f.relative_to(cache)}  ({size:,} bytes)")

    print(f"\nFinal loss: {losses[-1]:.6f}")
    print("Done.")


if __name__ == "__main__":
    main()
