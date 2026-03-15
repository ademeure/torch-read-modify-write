#!/usr/bin/env python3

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _nanochat_repo_dir() -> Path:
    return _repo_root() / "outputs" / "repos" / "nanochat"


def _default_nanochat_base_dir() -> Path:
    override = os.environ.get("NANOCHAT_BASE_DIR")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "nanochat"


def _prepare_nanochat_base_dir(base_dir: Path) -> None:
    """Link shared tokenizer/data into an isolated base dir for smoke tests."""
    source_base = _default_nanochat_base_dir()
    tokenizer_dir = source_base / "tokenizer"
    # nanochat switched from base_data to base_data_climbmix in March 2026
    data_dir = source_base / "base_data_climbmix"
    if not data_dir.exists():
        data_dir = source_base / "base_data"

    if not tokenizer_dir.exists():
        pytest.skip(f"nanochat tokenizer cache not found: {tokenizer_dir}")
    if not data_dir.exists():
        pytest.skip(f"nanochat data cache not found: {data_dir}")

    base_dir.mkdir(parents=True, exist_ok=True)
    os.symlink(tokenizer_dir, base_dir / "tokenizer", target_is_directory=True)
    os.symlink(data_dir, base_dir / data_dir.name, target_is_directory=True)


def test_nanochat_install_cli_smoke(tmp_path):
    repo_dir = _nanochat_repo_dir()
    if not (repo_dir / "nanochat" / "gpt.py").exists():
        pytest.skip("nanochat repo not found")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for nanochat CLI smoke test")

    cache_dir = tmp_path / ".torch_graph_cache"
    base_dir = tmp_path / "nanochat_base"
    _prepare_nanochat_base_dir(base_dir)

    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{_repo_root()}{os.pathsep}{pythonpath}"
        if pythonpath
        else str(_repo_root())
    )
    env["NANOCHAT_BASE_DIR"] = str(base_dir)
    env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    env.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("TRANSFORMERS_VERBOSITY", "error")

    cmd = [
        sys.executable,
        "-m",
        "torch_graph",
        "install",
        "--graph",
        "--cache-dir",
        str(cache_dir),
        "-m",
        "scripts.base_train",
        "--run",
        "dummy",
        "--model-tag",
        "pytest_smoke",
        "--depth",
        "4",
        "--max-seq-len",
        "512",
        "--device-batch-size",
        "1",
        "--total-batch-size",
        "512",
        "--num-iterations",
        "1",
        "--eval-every",
        "1",
        "--eval-tokens",
        "1024",
        "--core-metric-every",
        "-1",
        "--sample-every",
        "-1",
        "--save-every",
        "-1",
    ]
    result = subprocess.run(
        cmd,
        cwd=repo_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    output = result.stdout + "\n" + result.stderr

    assert result.returncode == 0, output
    assert "Validation bpb" in output
    assert "Capturing aten for GPT variant '2a_loss_reduction_eval'" in output
    assert "Capturing aten for GPT variant '2a_train'" in output

    assert list(cache_dir.glob("GPT_*_2a_loss_reduction_eval_aten.py"))
    assert list(cache_dir.glob("GPT_*_2a_loss_reduction_eval_aten.html"))
    assert list(cache_dir.glob("GPT_*_2a_train_aten.py"))
    assert list(cache_dir.glob("GPT_*_2a_train_aten.html"))
    # MuonAdamW uses inner @torch.compile functions (adamw_step_fused etc.)
    # so there's no monolithic optimizer aten file — inner fns are captured
    # individually via _CompiledFnProxy.
    inner_fn_files = list(cache_dir.glob("*_aten.py"))
    has_inner_fn = any(
        "step_fused" in f.name.lower() or "adamw" in f.name.lower()
        for f in inner_fn_files
        if "GPT" not in f.name and "optimizer_MuonAdamW" not in f.name
    )
    has_monolithic = bool(list(cache_dir.glob("optimizer_MuonAdamW_*_aten.py")))
    assert has_inner_fn or has_monolithic, \
        f"Expected inner fn or monolithic optimizer cache files, got: {[f.name for f in inner_fn_files]}"

    checkpoint_dir = base_dir / "base_checkpoints" / "pytest_smoke"
    assert (checkpoint_dir / "model_000001.pt").exists()
    assert (checkpoint_dir / "optim_000001_rank0.pt").exists()
