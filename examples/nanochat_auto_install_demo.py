#!/usr/bin/env python3
"""Demo: auto_install with nanochat — zero-manual-step torch.compile replacement.

This shows the full workflow:
1. import torch_graph.auto_install  (patches torch.compile)
2. Run nanochat's normal training loop
3. torch.compile calls are intercepted → aten graphs captured → installed
4. On second run, cached aten files are loaded from disk
5. User can edit the cached .py files to add custom CUDA/Triton kernels

Usage:
    python examples/nanochat_auto_install_demo.py              # first run: captures
    python examples/nanochat_auto_install_demo.py              # second run: loads cache
    python examples/nanochat_auto_install_demo.py --recapture  # force recapture
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Parse args ────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--recapture", action="store_true", help="Force recapture even if cache exists")
parser.add_argument("--steps", type=int, default=3, help="Training steps")
args = parser.parse_args()

# ══════════════════════════════════════════════════════════════════════
# Step 0: Patch torch.compile BEFORE importing anything that uses it
# ══════════════════════════════════════════════════════════════════════

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "outputs", "nanochat_aten_cache")

import torch_graph.auto_install
torch_graph.auto_install.configure(
    cache_dir=CACHE_DIR,
    verbose=True,
    force_recapture=args.recapture,
    capture_backward=True,
    num_real_outputs=1,
)

import torch
torch.compiler.reset()  # ensure clean compiler state across runs

# ── Setup nanochat ────────────────────────────────────────────────────

REPO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "outputs", "repos", "nanochat")

def ensure_repo():
    if not os.path.exists(os.path.join(REPO_DIR, "nanochat", "gpt.py")):
        import subprocess
        print(f"Cloning nanochat into {REPO_DIR}...")
        subprocess.check_call(
            ["git", "clone", "--depth", "1",
             "https://github.com/karpathy/nanochat.git", REPO_DIR],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

ensure_repo()
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)


# ══════════════════════════════════════════════════════════════════════
# Step 1: Build model + optimizer (exactly like nanochat)
# ══════════════════════════════════════════════════════════════════════

print("=" * 70)
print("STEP 1: Build nanochat model + optimizer")
print("=" * 70)

import tiktoken
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import SPECIAL_TOKENS, RustBPETokenizer
from nanochat.optim import MuonAdamW

# Tokenizer
base = tiktoken.get_encoding("gpt2")
special_tokens = {**base._special_tokens}
offset = base.n_vocab
for i, name in enumerate(SPECIAL_TOKENS):
    special_tokens[name] = offset + i
enc = tiktoken.Encoding(
    name="gpt2_nanochat",
    pat_str=base._pat_str,
    mergeable_ranks=base._mergeable_ranks,
    special_tokens=special_tokens,
)
tokenizer = RustBPETokenizer(enc, "<|bos|>")
vocab_size = tokenizer.get_vocab_size()

# Model (small for demo)
DEPTH = 4
ASPECT_RATIO = 32
HEAD_DIM = 64
SEQ_LEN = 32
BATCH_SIZE = 2

base_dim = DEPTH * ASPECT_RATIO
model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
num_heads = model_dim // HEAD_DIM

config = GPTConfig(
    n_layer=DEPTH,
    n_head=num_heads,
    n_kv_head=num_heads,
    n_embd=model_dim,
    vocab_size=vocab_size,
    sequence_len=SEQ_LEN,
    window_pattern="S" * DEPTH,
)

torch.manual_seed(42)
with torch.device("cpu"):
    model = GPT(config, pad_vocab_size_to=64)
model.init_weights()
model.train()

print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
print(f"Config: depth={DEPTH}, dim={model_dim}, heads={num_heads}, seq_len={SEQ_LEN}")


# Optimizer
def build_optimizer(model):
    param_groups = []
    muon_groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2:
            param_groups.append({
                "params": [p], "kind": "adamw",
                "lr": 0.004, "betas": (0.8, 0.95), "eps": 1e-8, "weight_decay": 0.0,
            })
        elif "wte" in name or "lm_head" in name or "value_embeds" in name:
            lr = 0.3 if ("wte" in name or "value_embeds" in name) else 0.004
            param_groups.append({
                "params": [p], "kind": "adamw",
                "lr": lr, "betas": (0.8, 0.95), "eps": 1e-8, "weight_decay": 0.0,
            })
        else:
            key = str(p.shape)
            if key not in muon_groups:
                muon_groups[key] = []
            muon_groups[key].append(p)
    for key, params in muon_groups.items():
        param_groups.append({
            "params": params, "kind": "muon",
            "lr": 0.02, "momentum": 0.95, "ns_steps": 5,
            "beta2": 0.8, "weight_decay": 0.0,
        })
    return MuonAdamW(param_groups)


# Sample data
texts = [
    "The Pythagorean theorem states that in a right-angled triangle, "
    "the square of the hypotenuse is equal to the sum of the squares "
    "of the other two sides.",
    "Machine learning is a subset of artificial intelligence.",
    "Paris is the capital of France. Berlin is the capital of Germany.",
    "Water freezes at zero degrees Celsius.",
]
token_pool = []
for text in texts:
    token_pool.extend(tokenizer.encode(text))


def get_batch(step):
    torch.manual_seed(1337 + step)
    pool_len = len(token_pool)
    offsets = torch.randint(0, pool_len - SEQ_LEN - 1, (BATCH_SIZE,))
    rows = []
    for off in offsets:
        start = off.item()
        rows.append(token_pool[start:start + SEQ_LEN + 1])
    batch = torch.tensor(rows, dtype=torch.long)
    x = batch[:, :-1].contiguous()
    targets = batch[:, 1:].contiguous()
    return x, targets


# ══════════════════════════════════════════════════════════════════════
# Step 2: This is where the magic happens — torch.compile is intercepted
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("STEP 2: torch.compile(model) → auto_install intercepts")
print("=" * 70)

torch.manual_seed(42)
optimizer = build_optimizer(model)

# This calls our patched torch.compile!
# On first run: captures aten graphs, saves to cache, installs via autograd.Function
# On second run: loads from cache (or user-modified .py file)
compiled_model = torch.compile(model, dynamic=False)

print(f"Type of 'compiled' model: {type(compiled_model).__name__}")
print(f"  (It's a proxy — first forward call triggers capture/install)")


# ══════════════════════════════════════════════════════════════════════
# Step 3: Train! (first forward call triggers the capture)
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print(f"STEP 3: Training for {args.steps} steps")
print("=" * 70)

for step in range(args.steps):
    x, targets = get_batch(step)
    loss = compiled_model(x, targets=targets)
    loss.backward()
    optimizer.step()
    model.zero_grad(set_to_none=True)
    print(f"  step {step}: loss = {loss.item():.6f}")

print(f"\nFinal loss: {loss.item():.6f}")


# ══════════════════════════════════════════════════════════════════════
# Step 4: Show what's in the cache
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("STEP 4: Cache status")
print("=" * 70)

print(torch_graph.auto_install.status())

print(f"\nCache directory: {CACHE_DIR}")
if os.path.exists(CACHE_DIR):
    for f in sorted(os.listdir(CACHE_DIR)):
        fpath = os.path.join(CACHE_DIR, f)
        size = os.path.getsize(fpath)
        print(f"  {f}  ({size:,} bytes)")

print(f"""
═══════════════════════════════════════════════════════════════════════
HOW TO USE:

1. The aten .py file in {CACHE_DIR} contains the
   full forward+backward computation graph as editable Python code.

2. Edit ANY operation in that file — replace aten ops with custom
   CUDA kernels, Triton kernels, or anything callable from Python.

3. Re-run this script — it will detect your edits and load the
   modified file instead of re-capturing.

4. The backward() function is also editable — you have full control
   over the gradient computation.

Example: Replace a matmul with a custom kernel:
   # Before (in the cached .py):
   mm = aten.mm(x, weight)

   # After (your edit):
   import my_triton_kernels
   mm = my_triton_kernels.custom_matmul(x, weight)
═══════════════════════════════════════════════════════════════════════
""")
