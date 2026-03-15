"""Recipes for karpathy/nanochat GPT model with real tokenizer and data.

Three variants:
  setup()          – base pre-training (MuonAdamW, causal LM loss)
  setup_sft()      – supervised fine-tuning (real conversation masking)
  setup_rl()       – RL / GRPO-style policy gradient loss

All variants use the real nanochat tokenizer (GPT-2 BPE + 9 special tokens)
and real conversation data rendered through the tokenizer pipeline, exactly
as the actual training scripts (base_train.py, chat_sft.py, chat_rl.py) do.

Expects the repo at ``outputs/repos/nanochat`` (the CLI will clone it if missing).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

_REPO_DIR = Path(__file__).resolve().parent.parent / "outputs" / "repos" / "nanochat"

# ── Sample conversations (used in SFT and RL recipes) ────────────────

# Multi-turn conversations in nanochat's format.  These exercise
# all token types: user, assistant, tool calls, tool output, system.
CONVERSATIONS = [
    # Simple Q&A
    {
        "messages": [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
        ]
    },
    # Multi-turn
    {
        "messages": [
            {"role": "user", "content": "Solve 15 * 23"},
            {"role": "assistant", "content": "15 times 23 equals 345."},
            {"role": "user", "content": "Now divide that by 5"},
            {"role": "assistant", "content": "345 divided by 5 is 69."},
        ]
    },
    # Tool use (python REPL)
    {
        "messages": [
            {"role": "user", "content": "How many letters are in strawberry?"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Let me count: "},
                {"type": "python", "text": 'len("strawberry")'},
                {"type": "python_output", "text": "10"},
                {"type": "text", "text": "There are 10 letters in strawberry."},
            ]},
        ]
    },
    # System message + longer response
    {
        "messages": [
            {"role": "system", "content": "You are a helpful math tutor."},
            {"role": "user", "content": "Explain the Pythagorean theorem"},
            {"role": "assistant", "content": (
                "The Pythagorean theorem states that in a right triangle, "
                "the square of the hypotenuse equals the sum of the squares "
                "of the other two sides: a squared plus b squared equals c squared."
            )},
        ]
    },
    # Coding help with tool
    {
        "messages": [
            {"role": "user", "content": "Write a function to reverse a string in Python"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Here is a simple approach:\n"},
                {"type": "python", "text": "def reverse(s): return s[::-1]\nreverse('hello')"},
                {"type": "python_output", "text": "'olleh'"},
                {"type": "text", "text": "The function uses Python slice notation to reverse."},
            ]},
        ]
    },
    # Short factual
    {
        "messages": [
            {"role": "user", "content": "What year did the Berlin Wall fall?"},
            {"role": "assistant", "content": "The Berlin Wall fell in 1989."},
        ]
    },
]


# ── Helpers ───────────────────────────────────────────────────────────

def _ensure_repo():
    if not (_REPO_DIR / "nanochat" / "gpt.py").exists():
        print(f"Cloning nanochat into {_REPO_DIR} …")
        subprocess.check_call(
            ["git", "clone", "--depth", "1",
             "https://github.com/karpathy/nanochat.git", str(_REPO_DIR)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )


def _init():
    """Ensure repo is cloned and nanochat is on sys.path."""
    _ensure_repo()
    if str(_REPO_DIR) not in sys.path:
        sys.path.insert(0, str(_REPO_DIR))


def _build_tokenizer():
    """Build the real nanochat tokenizer (GPT-2 BPE + 9 special tokens)."""
    import tiktoken
    from nanochat.tokenizer import SPECIAL_TOKENS, RustBPETokenizer

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
    return RustBPETokenizer(enc, "<|bos|>")


def _build_model(vocab_size: int, depth: int = 4, aspect_ratio: int = 32,
                  head_dim: int = 64, seq_len: int = 128):
    """Build a nanochat GPT model with the given vocab size and scale."""
    from nanochat.gpt import GPT, GPTConfig

    base_dim = depth * aspect_ratio
    model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
    num_heads = model_dim // head_dim

    config = GPTConfig(
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        vocab_size=vocab_size,
        sequence_len=seq_len,
        window_pattern="S" * depth,
    )

    with torch.device("cpu"):
        model = GPT(config, pad_vocab_size_to=64)
    model.init_weights()
    model.train()
    return model, config


def _build_optimizer(model):
    """Build MuonAdamW optimizer with nanochat parameter grouping.

    Mirrors base_train.py:
    - Embedding/unembedding params → AdamW
    - Scalar params (lambdas) → AdamW with per-type LR
    - 2D matrix params → Muon (grouped by shape for stacking)
    """
    from nanochat.optim import MuonAdamW

    param_groups = []
    muon_groups = {}  # shape_key → list of params

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if p.ndim < 2:
            if "resid_lambdas" in name:
                lr = 0.005
            elif "x0_lambdas" in name:
                lr = 0.5
            else:
                lr = 0.004
            param_groups.append({
                "params": [p], "kind": "adamw",
                "lr": lr, "betas": (0.8, 0.95), "eps": 1e-8, "weight_decay": 0.0,
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


# ── Pre-training recipe ──────────────────────────────────────────────

def setup() -> dict:
    """Base pre-training: causal LM loss with MuonAdamW.

    Uses the real tokenizer to produce token IDs from real text.
    Mirrors base_train.py's training loop.
    """
    _init()
    tokenizer = _build_tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    model, config = _build_model(vocab_size)
    optimizer = _build_optimizer(model)

    _token_pool = _make_token_pool(tokenizer)

    batch_size = 2
    seq_len = 64

    def get_batch(step: int):
        torch.manual_seed(1337 + step)
        # Sample from real token pool (with wrapping)
        pool_len = len(_token_pool)
        offsets = torch.randint(0, pool_len - seq_len - 1, (batch_size,))
        rows = []
        for off in offsets:
            start = off.item()
            rows.append(_token_pool[start:start + seq_len + 1])
        batch = torch.tensor(rows, dtype=torch.long)
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


# ── SFT recipe ───────────────────────────────────────────────────────

def setup_sft() -> dict:
    """SFT: real conversation masking via nanochat's tokenizer.

    Uses the real render_conversation() pipeline from nanochat/tokenizer.py:
    - BOS token at start
    - <|user_start|> ... <|user_end|> with mask=0 (not trained on)
    - <|assistant_start|> ... <|assistant_end|> with mask=1 (trained on)
    - <|python_start|>/<|python_end|> for tool calls (mask=1)
    - <|output_start|>/<|output_end|> for tool output (mask=0)
    - Targets = shifted IDs; masked positions get targets=-1
    """
    _init()
    tokenizer = _build_tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    model, config = _build_model(vocab_size)
    optimizer = _build_optimizer(model)

    seq_len = config.sequence_len

    # Pre-render all conversations through the real tokenizer pipeline
    _rendered = []
    for conv in CONVERSATIONS:
        ids, mask = tokenizer.render_conversation(conv, max_tokens=seq_len + 1)
        _rendered.append((ids, mask))

    batch_size = 2

    def get_batch(step: int):
        # Pick conversations for this batch (cycling through pool)
        rows_ids, rows_masks = [], []
        for b in range(batch_size):
            idx = (step * batch_size + b) % len(_rendered)
            ids, mask = _rendered[idx]
            # Pad to seq_len+1 if needed (BOS padding, mask=0)
            bos = tokenizer.get_bos_token_id()
            if len(ids) < seq_len + 1:
                pad_len = seq_len + 1 - len(ids)
                ids = ids + [bos] * pad_len
                mask = mask + [0] * pad_len
            rows_ids.append(ids[:seq_len + 1])
            rows_masks.append(mask[:seq_len + 1])

        batch = torch.tensor(rows_ids, dtype=torch.long)
        mask_tensor = torch.tensor(rows_masks, dtype=torch.int8)

        # SFT data format (mirrors chat_sft.py lines 285-299)
        inputs = batch[:, :-1].to(dtype=torch.int32).contiguous()
        targets = batch[:, 1:].to(dtype=torch.int64).contiguous()
        # Apply mask: positions where mask=0 get targets=-1 (ignore_index)
        mask_targets = mask_tensor[:, 1:].contiguous()
        targets[mask_targets == 0] = -1

        return (inputs,), {"targets": targets}

    sample_args, sample_kwargs = get_batch(0)

    return {
        "model": model,
        "sample_args": sample_args,
        "sample_kwargs": sample_kwargs,
        "get_batch": get_batch,
        "optimizer": optimizer,
    }


# ── RL recipe ────────────────────────────────────────────────────────

def setup_rl() -> dict:
    """RL (GRPO/REINFORCE): per-token policy gradient loss.

    Mirrors chat_rl.py:
      logp = -model(inputs, targets, loss_reduction='none')
      pg_obj = (logp * advantages).sum() / num_valid_tokens
      loss = -pg_obj

    Uses real tokenized conversations as "rollout" sequences.
    Prompt tokens are masked (targets=-1), only completions contribute.
    """
    _init()
    tokenizer = _build_tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    model, config = _build_model(vocab_size)
    optimizer = _build_optimizer(model)

    seq_len = config.sequence_len
    batch_size = 2  # simulates num_samples rollouts per example

    # Pre-render conversations (simulate completed rollouts)
    _rendered = []
    for conv in CONVERSATIONS:
        ids, mask = tokenizer.render_conversation(conv, max_tokens=seq_len + 1)
        _rendered.append((ids, mask))

    # Simulated rewards for each "rollout" (like GSM8K correct/incorrect)
    torch.manual_seed(42)
    _rewards = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])

    def _rl_loss_fn(per_token_nll):
        """GRPO-style policy gradient loss from per-token NLL."""
        logp = -per_token_nll.view(batch_size, seq_len)
        # Advantages = reward - mean(reward) (DAPO-style, token-level)
        step_rewards = _rewards[:batch_size].to(logp.device)
        advantages = step_rewards - step_rewards.mean()
        pg_obj = (logp * advantages.unsqueeze(-1)).sum()
        # Normalize by number of valid (non-masked) tokens
        num_valid = (per_token_nll != 0).sum().clamp(min=1)
        pg_obj = pg_obj / num_valid
        return -pg_obj

    def get_batch(step: int):
        rows_ids = []
        for b in range(batch_size):
            idx = (step * batch_size + b) % len(_rendered)
            ids, mask = _rendered[idx]
            bos = tokenizer.get_bos_token_id()
            if len(ids) < seq_len + 1:
                pad_len = seq_len + 1 - len(ids)
                ids = ids + [bos] * pad_len
                mask = mask + [0] * pad_len
            rows_ids.append(ids[:seq_len + 1])

        batch = torch.tensor(rows_ids, dtype=torch.long)
        inputs = batch[:, :-1].to(dtype=torch.int32).contiguous()
        targets = batch[:, 1:].to(dtype=torch.int64).contiguous()

        # RL masking: mask out prompt tokens (find where assistant starts)
        # For simplicity, use render_for_completion to get prompt length,
        # then mask everything before it
        for b in range(batch_size):
            idx = (step * batch_size + b) % len(_rendered)
            _, conv_mask = _rendered[idx]
            conv_mask = conv_mask[:seq_len + 1]
            mask_t = torch.tensor(conv_mask[1:seq_len + 1], dtype=torch.int8)
            # Pad mask if needed
            if len(mask_t) < seq_len:
                mask_t = torch.cat([mask_t, torch.zeros(seq_len - len(mask_t), dtype=torch.int8)])
            targets[b][mask_t == 0] = -1

        return (inputs,), {"targets": targets, "loss_reduction": "none"}

    sample_args, sample_kwargs = get_batch(0)

    return {
        "model": model,
        "sample_args": sample_args,
        "sample_kwargs": sample_kwargs,
        "get_batch": get_batch,
        "optimizer": optimizer,
        "loss_fn": _rl_loss_fn,
    }


# ── Scaled recipes ────────────────────────────────────────────────────

def _make_token_pool(tokenizer, min_tokens=512):
    """Build a token pool with enough tokens for batching."""
    texts = [
        "The Pythagorean theorem states that in a right-angled triangle, "
        "the square of the hypotenuse equals the sum of the squares of the other two sides.",
        "Machine learning allows systems to learn and improve from experience "
        "without being explicitly programmed. Deep learning uses neural networks.",
        "Paris is the capital of France. Berlin is the capital of Germany. "
        "Tokyo is the capital of Japan. Canberra is the capital of Australia.",
        "To reverse a string in Python, you can use slice notation: s[::-1]. "
        "This creates a new string with characters in reverse order.",
        "The speed of light in a vacuum is approximately 299,792,458 meters "
        "per second. This is a fundamental constant in physics.",
        "Water freezes at zero degrees Celsius and boils at one hundred "
        "degrees Celsius at standard atmospheric pressure.",
    ]
    pool = []
    while len(pool) < min_tokens:
        for text in texts:
            pool.extend(tokenizer.encode(text))
    return pool


def setup_12layer() -> dict:
    """12-layer model (~176M params). depth=12, aspect_ratio=32, head_dim=64 → 384-dim, 6 heads."""
    _init()
    tokenizer = _build_tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    model, config = _build_model(vocab_size, depth=12, aspect_ratio=32, head_dim=64, seq_len=128)
    optimizer = _build_optimizer(model)

    _token_pool = _make_token_pool(tokenizer)
    batch_size, seq_len = 2, 64

    def get_batch(step: int):
        torch.manual_seed(1337 + step)
        pool_len = len(_token_pool)
        offsets = torch.randint(0, pool_len - seq_len - 1, (batch_size,))
        rows = [_token_pool[o.item():o.item() + seq_len + 1] for o in offsets]
        batch = torch.tensor(rows, dtype=torch.long)
        x = batch[:, :-1].contiguous()
        targets = batch[:, 1:].contiguous()
        return (x,), {"targets": targets}

    sample_args, sample_kwargs = get_batch(0)
    return {
        "model": model, "sample_args": sample_args, "sample_kwargs": sample_kwargs,
        "get_batch": get_batch, "optimizer": optimizer,
    }


def setup_20layer() -> dict:
    """20-layer model (~1.2B params). depth=20, aspect_ratio=64, head_dim=128 → 1280-dim, 10 heads."""
    _init()
    tokenizer = _build_tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    model, config = _build_model(vocab_size, depth=20, aspect_ratio=64, head_dim=128, seq_len=256)
    optimizer = _build_optimizer(model)

    _token_pool = _make_token_pool(tokenizer)
    batch_size, seq_len = 1, 128

    def get_batch(step: int):
        torch.manual_seed(1337 + step)
        pool_len = len(_token_pool)
        offsets = torch.randint(0, pool_len - seq_len - 1, (batch_size,))
        rows = [_token_pool[o.item():o.item() + seq_len + 1] for o in offsets]
        batch = torch.tensor(rows, dtype=torch.long)
        x = batch[:, :-1].contiguous()
        targets = batch[:, 1:].contiguous()
        return (x,), {"targets": targets}

    sample_args, sample_kwargs = get_batch(0)
    return {
        "model": model, "sample_args": sample_args, "sample_kwargs": sample_kwargs,
        "get_batch": get_batch, "optimizer": optimizer,
    }
