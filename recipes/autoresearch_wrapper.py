"""Recipes for karpathy/autoresearch GPT model.

Three variants:
  setup()         – base pre-training (MuonAdamW, causal LM loss)
  setup_small()   – smaller model (depth=4) for quick testing
  setup_eval()    – eval-only (no optimizer, no backward)

Expects the repo at ``outputs/repos/autoresearch``.
Uses real tokenizer and data from ~/.cache/autoresearch/ (run prepare.py first).

NOTE: autoresearch/train.py has no ``if __name__ == "__main__"`` guard — all
training code runs at import time.  We work around this by using Python's AST
module to parse the file and exec only class/function/import/constant
definitions, skipping the training loop entirely.  This means we can import
GPT, GPTConfig, MuonAdamW etc. without running a 5-minute training session.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import types
from pathlib import Path

import torch

_REPO_DIR = Path(__file__).resolve().parent.parent / "outputs" / "repos" / "autoresearch"
_train_ns: dict | None = None  # cached namespace from partial exec of train.py


# ── Helpers ───────────────────────────────────────────────────────────

def _ensure_repo():
    if not (_REPO_DIR / "train.py").exists():
        print(f"Cloning autoresearch into {_REPO_DIR} …")
        _REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(
            ["git", "clone", "--depth", "1",
             "https://github.com/karpathy/autoresearch.git", str(_REPO_DIR)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )


def _init():
    """Ensure repo is cloned and autoresearch is on sys.path."""
    _ensure_repo()
    if str(_REPO_DIR) not in sys.path:
        sys.path.insert(0, str(_REPO_DIR))


def _load_train_module():
    """Load autoresearch/train.py using AST surgery to skip the training loop.

    Parses train.py into an AST, keeps only:
      - imports
      - class/function definitions (GPT, MuonAdamW, etc.)
      - simple assignments (constants, dataclass decorators)
      - decorated definitions (@torch.compile, @dataclass)

    Drops all bare expressions and statements that constitute the
    training loop (everything after the hyperparameter block).
    """
    global _train_ns
    if _train_ns is not None:
        return _train_ns

    _init()
    train_path = _REPO_DIR / "train.py"
    source = train_path.read_text()
    tree = ast.parse(source, str(train_path))

    def _target_names(node):
        names = []
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.append(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.append(node.target.id)
        return names

    # Keep only "safe" top-level nodes
    safe_nodes = []
    for node in tree.body:
        # The real training script starts here. Everything below this point is
        # runtime setup / execution, not definitions we want to import.
        if any(name in {"config", "model", "optimizer", "train_loader"} for name in _target_names(node)):
            break
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            safe_nodes.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            safe_nodes.append(node)
        elif isinstance(node, ast.Assign):
            # Keep simple constant assignments (ASPECT_RATIO = 64, etc.)
            # but skip things like `model = torch.compile(model)` or
            # `x, y, epoch = next(train_loader)` which are training loop
            safe_nodes.append(node)
        elif isinstance(node, ast.AnnAssign):
            safe_nodes.append(node)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            # Module docstring
            safe_nodes.append(node)
        # Skip: ast.Expr (bare function calls), ast.While, ast.For, ast.If
        # (unless it's an if __name__ guard), ast.Assert, etc.

    tree.body = safe_nodes
    ast.fix_missing_locations(tree)
    code = compile(tree, str(train_path), "exec")

    # Register as a real module before exec so decorators like @dataclass
    # can resolve cls.__module__ through sys.modules during execution.
    mod = types.ModuleType("train")
    mod.__file__ = str(train_path)
    mod.__dict__.update({"__name__": "train", "__file__": str(train_path)})
    sys.modules["train"] = mod

    exec(code, mod.__dict__)

    _train_ns = mod.__dict__
    return _train_ns


def _build_model(depth: int = 4, aspect_ratio: int = 64, head_dim: int = 128,
                  seq_len: int = 64, window_pattern: str = "SSSL",
                  device: str = "cuda"):
    """Build an autoresearch GPT model."""
    ns = _load_train_module()
    from prepare import Tokenizer

    GPT = ns["GPT"]
    GPTConfig = ns["GPTConfig"]

    tokenizer = Tokenizer.from_directory()
    vocab_size = tokenizer.get_vocab_size()

    base_dim = depth * aspect_ratio
    model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
    num_heads = model_dim // head_dim

    config = GPTConfig(
        sequence_len=seq_len,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern=window_pattern,
    )

    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=torch.device(device))
    model.init_weights()
    model.train()

    return model, config, tokenizer


def _build_optimizer(model):
    """Build MuonAdamW optimizer via model.setup_optimizer()."""
    return model.setup_optimizer()


def _make_token_pool(tokenizer, seq_len, min_tokens=2048):
    """Build a token pool from real data via prepare.py's dataloader."""
    _init()
    from prepare import make_dataloader

    loader = make_dataloader(tokenizer, 1, seq_len, "val")
    tokens = []
    while len(tokens) < min_tokens:
        x, y, _ = next(loader)
        tokens.extend(x[0].tolist())
        tokens.extend([y[0, -1].item()])
    return tokens


# ── Pre-training recipe ──────────────────────────────────────────────

def setup(device="cuda") -> dict:
    """Base pre-training: causal LM loss with MuonAdamW."""
    model, config, tokenizer = _build_model(depth=4, seq_len=64, device=device)
    optimizer = _build_optimizer(model)

    batch_size = 2
    seq_len = 64
    _token_pool = _make_token_pool(tokenizer, seq_len)

    def get_batch(step: int):
        torch.manual_seed(1337 + step)
        pool_len = len(_token_pool)
        offsets = torch.randint(0, pool_len - seq_len - 1, (batch_size,))
        rows = [_token_pool[o.item():o.item() + seq_len + 1] for o in offsets]
        batch = torch.tensor(rows, dtype=torch.long, device=device)
        x = batch[:, :-1].contiguous()
        targets = batch[:, 1:].contiguous()
        return (x,), {"targets": targets}

    sample_args, sample_kwargs = get_batch(0)

    return {
        "model": model,
        "sample_args": sample_args,
        "sample_kwargs": sample_kwargs,
        "get_batch": get_batch,
        "optimizer": optimizer,
    }


def setup_small(device="cuda") -> dict:
    """Small model (depth=2) for quick testing."""
    model, config, tokenizer = _build_model(depth=2, aspect_ratio=32,
                                             head_dim=64, seq_len=32,
                                             window_pattern="SL",
                                             device=device)
    batch_size = 2
    seq_len = 32
    _token_pool = _make_token_pool(tokenizer, seq_len)

    def get_batch(step: int):
        torch.manual_seed(1337 + step)
        pool_len = len(_token_pool)
        offsets = torch.randint(0, pool_len - seq_len - 1, (batch_size,))
        rows = [_token_pool[o.item():o.item() + seq_len + 1] for o in offsets]
        batch = torch.tensor(rows, dtype=torch.long, device=device)
        return (batch[:, :-1].contiguous(),), {"targets": batch[:, 1:].contiguous()}

    sample_args, sample_kwargs = get_batch(0)
    return {
        "model": model,
        "sample_args": sample_args,
        "sample_kwargs": sample_kwargs,
        "get_batch": get_batch,
    }


def setup_eval(device="cuda") -> dict:
    """Eval-only (no optimizer, no backward)."""
    model, config, tokenizer = _build_model(depth=4, seq_len=64, device=device)
    model.eval()

    batch_size = 2
    seq_len = 64
    _token_pool = _make_token_pool(tokenizer, seq_len)

    torch.manual_seed(42)
    pool_len = len(_token_pool)
    offsets = torch.randint(0, pool_len - seq_len - 1, (batch_size,))
    rows = [_token_pool[o.item():o.item() + seq_len + 1] for o in offsets]
    batch = torch.tensor(rows, dtype=torch.long, device=device)
    x = batch[:, :-1].contiguous()

    return {
        "model": model,
        "sample_args": (x,),
    }
