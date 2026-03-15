"""Recipe for KellerJordan/modded-nanogpt GPT-2 speedrun model.

Captures the full model with:
  - FP8 custom matmul ops (nanogpt::mm_t, nanogpt::mm_t_backward)
  - Flash Attention 3 varlen (Hopper)
  - Triton fused kernels (ReLU^2 MLP, softcapped cross-entropy)
  - Polar Express orthogonalization (in NorMuon optimizer)

Requires CUDA (H100/Hopper for FA3).

Expects the repo at ``outputs/repos/modded-nanogpt``
(the recipe will clone it if missing).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import torch

_REPO_DIR = Path(__file__).resolve().parent.parent / "outputs" / "repos" / "modded-nanogpt"
_REPO_URL = "https://github.com/KellerJordan/modded-nanogpt.git"


def _ensure_repo():
    if not (_REPO_DIR / "train_gpt.py").exists():
        print(f"Cloning modded-nanogpt into {_REPO_DIR} …")
        _REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(
            ["git", "clone", "--depth", "1", _REPO_URL, str(_REPO_DIR)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )


def _load_train_module():
    """Load train_gpt.py definitions without running the training loop.

    Patches out:
    - File-reading preamble (reads sys.argv[0])
    - CUDA warmup torch.empty().backward()
    - dist.init_process_group (we init it ourselves)
    - dist.barrier
    - Everything after '# int main'
    """
    # Install custom op tracker BEFORE loading the module, so FA3 ops
    # registered during get_kernel() are captured by Hook 3.
    import torch_graph.custom_ops  # noqa: F401

    source = (_REPO_DIR / "train_gpt.py").read_text()

    # Truncate at "# int main" section
    marker = "# int main"
    idx = source.find(marker)
    if idx > 0:
        source = source[:idx]

    # Remove file-reading preamble (lines 4-9: reads sys.argv[0] and triton_kernels.py)
    source = re.sub(
        r"# Read the current file.*?code \+= f\.read\(\)\n",
        'code = ""\n',
        source,
        flags=re.DOTALL,
    )

    # Remove torch.empty().backward() CUDA warmup
    source = re.sub(
        r"torch\.empty\(\n\s+1.*?\.backward\(\).*?\n",
        "",
        source,
    )

    # Remove dist.init_process_group and dist.barrier
    source = source.replace(
        'dist.init_process_group(backend="cuda:nccl,cpu:gloo", device_id=device)\n',
        "",
    )
    source = source.replace("dist.barrier()\n", "")

    # Add repo dir to sys.path for imports (triton_kernels, kernels)
    if str(_REPO_DIR) not in sys.path:
        sys.path.insert(0, str(_REPO_DIR))

    # Create a real module object so @dataclass can resolve types via sys.modules
    import types
    mod = types.ModuleType("train_gpt")
    mod.__file__ = str(_REPO_DIR / "train_gpt.py")
    sys.modules["train_gpt"] = mod
    ns = mod.__dict__
    exec(compile(source, str(_REPO_DIR / "train_gpt.py"), "exec"), ns)

    return ns


def _init_distributed():
    """Initialize single-GPU distributed environment for modded-nanogpt."""
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")

    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=0, world_size=1)


def setup() -> dict:
    """Build modded-nanogpt model with synthetic data for capture.

    Returns the standard recipe dict with model, sample_args, sample_kwargs.
    The model's forward signature is:
        forward(input_seq, target_seq, seqlens, bigram_input_seq, schedule_cfg)
    """
    if not torch.cuda.is_available():
        raise RuntimeError("modded-nanogpt requires CUDA (H100/Hopper for FA3)")

    _ensure_repo()
    _init_distributed()

    ns = _load_train_module()
    GPT = ns["GPT"]
    ForwardScheduleConfig = ns["ForwardScheduleConfig"]
    get_bigram_hash = ns["get_bigram_hash"]
    device = ns["device"]

    # Create model (same hyperparameters as train_gpt.py)
    num_tokens = 1024  # Must be multiple of 16
    model = GPT(
        vocab_size=50257,
        num_layers=11,
        num_heads=6,
        head_dim=128,
        model_dim=768,
        max_seq_len=num_tokens,
    ).cuda()

    # Convert to bf16 (same as train_gpt.py lines 1880-1886)
    for m in model.modules():
        if isinstance(m, (torch.nn.Embedding, torch.nn.Linear)):
            m.weight.data = m.weight.data.bfloat16()
    model.attn_gate_bank.data = model.attn_gate_bank.data.bfloat16()
    model.ve_gate_bank.data = model.ve_gate_bank.data.bfloat16()
    model.attn_bank.data = model.attn_bank.data.bfloat16()
    model.mlp_bank.data = model.mlp_bank.data.bfloat16()

    model.train()

    # Create synthetic data
    torch.manual_seed(42)
    input_seq = torch.randint(0, 50257, (num_tokens,), dtype=torch.int32, device=device)
    target_seq = torch.randint(0, 50257, (num_tokens,), dtype=torch.int64, device=device)

    # Cumulative sequence lengths for varlen attention.
    # Format: sorted int32, starts at 0, ends at num_tokens.
    # Padding entries are set to num_tokens (zero-length "documents").
    next_multiple_of_n = ns["next_multiple_of_n"]
    max_num_docs = next_multiple_of_n(num_tokens // 300, n=128)
    seqlens = torch.full((max_num_docs,), num_tokens, dtype=torch.int32, device=device)
    seqlens[0] = 0
    # Two documents: [0, num_tokens//2, num_tokens, num_tokens, ...]
    seqlens[1] = num_tokens // 2

    # Bigram hash (computed on CPU, then transferred to CUDA)
    bigram_input_seq = get_bigram_hash(input_seq.cpu()).to(device=device)

    # Forward schedule config (stage 1 settings: block_size=128)
    schedule_cfg = ForwardScheduleConfig(
        mtp_weights=torch.tensor([1.0, 0.5, 0.25], device=device),
        ws_short=128,           # 1 * block_size
        ws_long=384,            # 3 * block_size
        train_max_seq_len=896,
    )

    sample_args = (input_seq, target_seq, seqlens, bigram_input_seq, schedule_cfg)

    result = {
        "model": model,
        "sample_args": sample_args,
        "sample_kwargs": {},
        "loss_fn": lambda out: out.sum(),
    }

    # Add optimizer + step_fn if requested (for full training loop capture)
    if os.environ.get("TORCH_GRAPH_CAPTURE_OPTIMIZER", ""):
        optimizer = setup_optimizer(model)
        result["optimizer"] = optimizer
        result["step_fn"] = lambda: optimizer.step(do_adam=True)

    return result


def setup_optimizer(model: torch.nn.Module):
    """Build NorMuonAndAdam optimizer matching train_gpt.py defaults.

    Must be called after ``setup()`` since it relies on the loaded train_gpt module.
    Returns the NorMuonAndAdam optimizer instance.
    """
    ns = sys.modules["train_gpt"].__dict__
    NorMuonAndAdam = ns["NorMuonAndAdam"]

    param_table = {
        "attn_bank":      {"optim": "normuon", "comms": "sharded",    "adam_betas": None},
        "mlp_bank":       {"optim": "normuon", "comms": "sharded",    "adam_betas": None},
        "scalars":        {"optim": "adam",    "comms": "replicated", "adam_betas": [0.9,  0.99], "lr_mul": 5.0,  "wd_mul": 0.0},
        "smear_gate":     {"optim": "adam",    "comms": "replicated", "adam_betas": [0.9,  0.99], "lr_mul": 0.01, "wd_mul": 0.0},
        "skip_gate":      {"optim": "adam",    "comms": "replicated", "adam_betas": [0.9,  0.99], "lr_mul": 0.05, "wd_mul": 0.0},
        "attn_gate_bank": {"optim": "adam",    "comms": "replicated", "adam_betas": [0.9,  0.99]},
        "ve_gate_bank":   {"optim": "adam",    "comms": "replicated", "adam_betas": [0.9,  0.99]},
        "lm_head":        {"optim": "adam",    "comms": "sharded",    "adam_betas": [0.5,  0.95], "wd_mul": 150.},
        "bigram_embed":   {"optim": "adam",    "comms": "sharded_sparse", "adam_betas": [0.75, 0.95], "lr_mul": 75.,  "wd_mul": 5.0},
        "post_lambdas":   {"optim": "adam",    "comms": "replicated",     "adam_betas": [0.9,  0.95], "lr_mul": 1.0,  "wd_mul": 0.0},
        "x0_lambdas":     {"optim": "adam",    "comms": "replicated",     "adam_betas": [0.9,  0.95], "lr_mul": 1.0,  "wd_mul": 0.0},
        "bigram_lambdas": {"optim": "adam",    "comms": "replicated",     "adam_betas": [0.9,  0.95], "lr_mul": 1.0,  "wd_mul": 0.0},
        "resid_lambdas":  {"optim": "adam",    "comms": "replicated",     "adam_betas": [0.9,  0.95], "lr_mul": 5.0,  "wd_mul": 0.0},
        "value_embeds":   {"optim": "adam",    "comms": "sharded",    "adam_betas": [0.75, 0.95], "lr_mul": 75.,  "wd_mul": 5.0},
        "embed":          {"optim": "adam",    "comms": "sharded",    "adam_betas": [0.5,  0.95], "wd_mul": 150.},
    }

    work_order = [
        "scalars", "smear_gate", "skip_gate", "attn_gate_bank", "ve_gate_bank",
        "post_lambdas", "x0_lambdas", "bigram_lambdas", "resid_lambdas",
        "value_embeds", "bigram_embed",
        "lm_head", "embed",
        "attn_bank", "mlp_bank",
    ]

    return NorMuonAndAdam(
        model.named_parameters(),
        param_table=param_table,
        scatter_order=list(param_table.keys()),
        work_order=work_order,
        adam_defaults=dict(lr=0.008, eps=1e-10, weight_decay=0.005),
        normuon_defaults=dict(lr=0.023, momentum=0.95, beta2=0.95, weight_decay=1.2),
    )
