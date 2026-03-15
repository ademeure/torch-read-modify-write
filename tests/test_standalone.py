"""Tests for standalone training loop generation.

Verifies that the generated standalone script produces the same losses
as running through auto_install's live replay.
"""

import copy
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_graph.auto_install as ai
from torch_graph.standalone import save_standalone_training


CACHE_DIR = Path(".torch_graph_cache/_test_standalone")


class MLP(nn.Module):
    """Simple MLP with loss in forward — matches test_multi_step.py."""

    def __init__(self, d=256, n=4):
        super().__init__()
        self.layers = nn.Sequential(*[nn.Linear(d, d) for _ in range(n)])
        self.head = nn.Linear(d, 10)

    def forward(self, x, target):
        h = self.layers(x)
        return F.cross_entropy(self.head(h), target)


@pytest.fixture(autouse=True)
def _cleanup():
    """Clean up caches and reset dynamo before each test."""
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
    torch._dynamo.reset()
    ai.unpatch()
    ai._installed.clear()
    ai._optimizer_captured.clear()
    yield
    ai.unpatch()
    ai._installed.clear()
    ai._optimizer_captured.clear()
    torch._dynamo.reset()
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)


def _parse_losses(stdout: str) -> list[float]:
    """Parse losses from standalone script stdout."""
    losses = []
    for line in stdout.strip().split("\n"):
        if line.startswith("Step "):
            loss_str = line.split("loss=")[1]
            losses.append(float(loss_str))
    return losses


def _capture_step1(model, x, target, *, dynamic=False, lr=1e-3):
    """Run 1 training step through auto_install, return (model, optimizer, loss)."""
    ai.configure(
        cache_dir=str(CACHE_DIR),
        capture_backward=True,
        replay_optimizer=True,
        force_recapture=True,
        verbose=False,
        dynamic=dynamic,
    )
    ai.patch()
    compiled = torch.compile(model, dynamic=dynamic)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    opt.zero_grad()
    loss = compiled(x, target)
    loss.backward()
    opt.step()
    return model, opt, loss.item()


def _capture_n_steps(model, x, target, n_steps, *, dynamic=False, lr=1e-3):
    """Run n_steps through auto_install with replay, return losses."""
    ai.configure(
        cache_dir=str(CACHE_DIR),
        capture_backward=True,
        replay_optimizer=True,
        force_recapture=True,
        verbose=False,
        dynamic=dynamic,
    )
    ai.patch()
    compiled = torch.compile(model, dynamic=dynamic)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    for _ in range(n_steps):
        opt.zero_grad()
        loss = compiled(x, target)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    ai.unpatch()
    return losses


def _run_standalone(script_path: Path) -> list[float]:
    """Run standalone script as subprocess, return parsed losses."""
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True, text=True, timeout=60,
    )
    if result.stderr:
        print(f"Standalone stderr:\n{result.stderr}")
    assert result.returncode == 0, (
        f"Standalone script failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    return _parse_losses(result.stdout)


# ---------------------------------------------------------------------------
# Monolithic AdamW tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_standalone_generation():
    """Verify standalone script can be generated from captured aten graphs."""
    torch.manual_seed(42)
    model = MLP().cuda()
    x = torch.randn(16, 256, device="cuda")
    target = torch.randint(10, (16,), device="cuda")

    model, opt, _ = _capture_step1(copy.deepcopy(model), x, target)

    script_path = save_standalone_training(
        model=model, optimizer=opt, cache_dir=CACHE_DIR, num_steps=3,
    )

    assert script_path.exists()
    content = script_path.read_text()
    assert "fw_bw.forward" in content
    assert "fw_bw.backward" in content
    assert "OPT_SLOTS" in content


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_standalone_matches_replay():
    """Standalone script produces same losses as live auto_install replay."""
    torch.manual_seed(42)
    model = MLP().cuda()
    x = torch.randn(16, 256, device="cuda")
    target = torch.randint(10, (16,), device="cuda")
    n_steps = 5

    # Reference: full replay
    replay_losses = _capture_n_steps(
        copy.deepcopy(model), x, target, n_steps,
    )

    # Re-capture 1 step for standalone generation
    ai._installed.clear()
    ai._optimizer_captured.clear()
    torch._dynamo.reset()

    model_s, opt_s, _ = _capture_step1(copy.deepcopy(model), x, target)

    script_path = save_standalone_training(
        model=model_s, optimizer=opt_s, cache_dir=CACHE_DIR,
        num_steps=n_steps - 1, sample_inputs=(x, target),
    )
    ai.unpatch()

    standalone_losses = _run_standalone(script_path)
    replay_post_capture = replay_losses[1:]

    assert len(standalone_losses) == len(replay_post_capture)
    for i, (sa, rp) in enumerate(zip(standalone_losses, replay_post_capture)):
        assert abs(sa - rp) < 1e-4, (
            f"Step {i}: standalone={sa:.6f} vs replay={rp:.6f} "
            f"(diff={abs(sa-rp):.2e})"
        )


# ---------------------------------------------------------------------------
# Dynamic shapes test
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_standalone_dynamic_shapes():
    """Standalone script works with dynamic shape captures (SymInt names)."""
    torch.manual_seed(42)
    model = MLP().cuda()
    x = torch.randn(16, 256, device="cuda")
    target = torch.randint(10, (16,), device="cuda")
    n_steps = 3

    # Reference: full replay with dynamic=True
    replay_losses = _capture_n_steps(
        copy.deepcopy(model), x, target, n_steps, dynamic=True,
    )

    ai._installed.clear()
    ai._optimizer_captured.clear()
    torch._dynamo.reset()

    model_s, opt_s, _ = _capture_step1(
        copy.deepcopy(model), x, target, dynamic=True,
    )

    script_path = save_standalone_training(
        model=model_s, optimizer=opt_s, cache_dir=CACHE_DIR,
        num_steps=n_steps - 1, sample_inputs=(x, target),
    )
    ai.unpatch()

    standalone_losses = _run_standalone(script_path)
    replay_post_capture = replay_losses[1:]

    assert len(standalone_losses) == len(replay_post_capture)
    for i, (sa, rp) in enumerate(zip(standalone_losses, replay_post_capture)):
        assert abs(sa - rp) < 1e-4, (
            f"Step {i}: standalone={sa:.6f} vs replay={rp:.6f}"
        )


# ---------------------------------------------------------------------------
# Autoresearch / MuonAdamW inner fn test
# ---------------------------------------------------------------------------


def _nanochat_available():
    """Check if nanochat is available. Returns (path, True) or (None, False)."""
    autoresearch_path = Path("outputs/repos/nanochat")
    if not autoresearch_path.exists():
        return None
    if not (autoresearch_path / "nanochat" / "gpt.py").exists():
        return None
    if not torch.cuda.is_available():
        return None
    if str(autoresearch_path) not in sys.path:
        sys.path.insert(0, str(autoresearch_path))
    return autoresearch_path


def _nanochat_model():
    """Create nanochat GPT model. Call BEFORE ai.patch() for model, AFTER for optimizer."""
    from nanochat.gpt import GPT, GPTConfig

    DEPTH = 2
    HEAD_DIM = 64
    base_dim = DEPTH * 32
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    vocab_size = 50304

    config = GPTConfig(
        n_layer=DEPTH, n_head=num_heads, n_kv_head=num_heads,
        n_embd=model_dim, vocab_size=vocab_size, sequence_len=16,
        window_pattern="S" * DEPTH,
    )

    torch.manual_seed(42)
    with torch.device("cpu"):
        model = GPT(config, pad_vocab_size_to=64)
    model.init_weights()
    model.train()
    model = model.cuda()

    idx = torch.randint(0, vocab_size, (2, 16), device="cuda")
    targets = torch.randint(0, vocab_size, (2, 16), device="cuda")

    return model, idx, targets, vocab_size, config


def _make_nanochat_optimizer(model):
    """Build MuonAdamW for a nanochat model. Must be called AFTER ai.patch()
    so that @torch.compile decorators in nanochat.optim get intercepted."""
    import importlib
    # Force reimport so @torch.compile decorators are intercepted by our patch
    import nanochat.optim
    importlib.reload(nanochat.optim)
    from nanochat.optim import MuonAdamW
    param_groups = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        param_groups.append({
            "params": [p], "kind": "adamw",
            "lr": 1e-3, "betas": (0.8, 0.95), "eps": 1e-8, "weight_decay": 0.0,
        })
    return MuonAdamW(param_groups)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_standalone_inner_fn_generation():
    """Verify standalone script is generated for inner fn optimizers."""
    if _nanochat_available() is None:
        pytest.skip("nanochat not available")

    model, idx, targets, vocab_size, config = _nanochat_model()
    model_s = copy.deepcopy(model)

    ai.configure(
        cache_dir=str(CACHE_DIR),
        capture_backward=True,
        replay_optimizer=True,
        force_recapture=True,
        verbose=False,
        dynamic=False,
    )
    ai.patch()

    # Optimizer must be created AFTER ai.patch() for inner fn detection
    opt_s = _make_nanochat_optimizer(model_s)
    compiled = torch.compile(model_s, dynamic=False)

    # Step 1: capture
    opt_s.zero_grad()
    loss = compiled(idx, targets=targets)
    loss.backward()
    opt_s.step()

    script_path = save_standalone_training(
        model=model_s, optimizer=opt_s, cache_dir=CACHE_DIR,
        num_steps=2, sample_inputs=(idx, targets),
    )
    ai.unpatch()

    assert script_path.exists()
    content = script_path.read_text()
    assert "inner_fns" in content
    assert "STEP_STATE" in content
    assert "OPT_SLOTS" not in content


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_standalone_inner_fn_matches_replay():
    """Inner fn standalone script produces same losses as live replay.

    Runs inline comparison: loads the standalone's aten modules and state
    directly, executes forward/backward/optimizer step by step, and compares
    against the live auto_install replay at every stage.
    """
    if _nanochat_available() is None:
        pytest.skip("nanochat not available")

    from torch_graph.standalone import _load_aten_module, _find_model_aten, _read_meta

    model, idx, targets, vocab_size, config = _nanochat_model()
    n_steps = 3

    # ── Single capture + live replay ──
    # Use ONE capture so standalone and live use the same aten modules.
    model_live = copy.deepcopy(model)
    ai.configure(
        cache_dir=str(CACHE_DIR), capture_backward=True,
        replay_optimizer=True, force_recapture=True,
        verbose=False, dynamic=False,
    )
    ai.patch()
    opt_live = _make_nanochat_optimizer(model_live)
    compiled_live = torch.compile(model_live, dynamic=False)

    # Step 1 (capture)
    opt_live.zero_grad()
    loss1 = compiled_live(idx, targets=targets)
    loss1.backward()
    opt_live.step()

    # Generate standalone state from the SAME capture (before any replay)
    save_standalone_training(
        model=model_live, optimizer=opt_live, cache_dir=CACHE_DIR,
        num_steps=n_steps - 1, sample_inputs=(idx, targets),
    )

    # Save state after step 1
    live_params_after1 = {
        n: p.detach().clone() for n, p in model_live.named_parameters()
    }

    # Steps 2+ (replay) - save reference values at each step
    live_diag = []
    for step_i in range(n_steps - 1):
        opt_live.zero_grad()
        loss = compiled_live(idx, targets=targets)
        loss.backward()
        grads = {
            n: p.grad.detach().clone()
            for n, p in model_live.named_parameters()
            if p.grad is not None
        }
        opt_live.step()
        params_after = {
            n: p.detach().clone() for n, p in model_live.named_parameters()
        }
        live_diag.append({
            "loss": loss.item(),
            "grads": grads,
            "params_after": params_after,
        })

    ai.unpatch()

    # ── Load standalone state + aten modules ──
    state = torch.load(
        CACHE_DIR / "standalone_state.pt",
        weights_only=False, map_location="cuda",
    )
    s_params = state["params"]
    s_opt_state = state["opt_state"]
    inner_fn_calls = state["inner_fn_calls"]
    group_params = state["group_params"]
    step_state_indices = state["step_state_indices"]
    sample_inputs = state["sample_inputs"]

    model_aten_path = _find_model_aten(CACHE_DIR)
    model_meta = _read_meta(model_aten_path)
    primal_names = model_meta["primal_names"]
    num_mutations = model_meta.get("num_mutations", 0)
    num_real_outputs = model_meta.get("num_real_outputs", 1)

    # Figure out user input positions (None entries in primal_names)
    user_positions = [
        i for i, name in enumerate(primal_names) if name is None
    ]

    # ── Compare initial params ──
    max_init_diff = 0
    for fw_pos, s_tensor in s_params.items():
        name = primal_names[fw_pos]
        if name in live_params_after1:
            diff = (s_tensor - live_params_after1[name]).abs().max().item()
            max_init_diff = max(max_init_diff, diff)
            if diff > 0:
                print(f"  Initial param DIFF {name}: {diff:.2e}")
    assert max_init_diff == 0, f"Initial params differ: max {max_init_diff:.2e}"

    fw_bw = _load_aten_module(model_aten_path)
    inner_fn_mods = {}
    for call in inner_fn_calls:
        fn_file = call["fn_file"]
        if fn_file not in inner_fn_mods:
            inner_fn_mods[fn_file] = _load_aten_module(CACHE_DIR / fn_file)

    # ── Run standalone steps inline, compare at each stage ──
    n_fw = len(primal_names)
    initial_step = state.get("initial_step", 1.0)
    _step_counter = initial_step

    for step_i in range(n_steps - 1):
        # Forward
        fw_in = [None] * n_fw
        for p, t in s_params.items():
            fw_in[p] = t
        for i, pos in enumerate(user_positions):
            if i < len(sample_inputs):
                fw_in[pos] = sample_inputs[i]

        result = fw_bw.forward(*fw_in)
        loss = result[num_mutations]
        saved = list(result[num_mutations + num_real_outputs:])
        non_t = [v for v in saved if not isinstance(v, torch.Tensor)]
        tens = [v for v in saved if isinstance(v, torch.Tensor)]
        saved = non_t + tens if non_t else saved

        # Backward
        grads = fw_bw.backward(*saved, torch.ones_like(loss))

        # Compare loss — tolerance grows per step due to bf16 accumulation drift
        loss_diff = abs(loss.item() - live_diag[step_i]["loss"])
        loss_tol = 1e-6 * (step_i + 1)
        assert loss_diff < loss_tol, (
            f"Step {step_i} loss diff: {loss_diff:.2e} "
            f"(standalone={loss.item():.8f} live={live_diag[step_i]['loss']:.8f})"
        )

        # Compare grads (step 0 should be exact; later steps may have
        # small diffs from cascading bf16 CUDA non-determinism in optimizer)
        max_grad_diff = 0
        for fw_pos in sorted(s_params.keys()):
            name = primal_names[fw_pos]
            if name in live_diag[step_i]["grads"]:
                s_grad = grads[fw_pos]
                if isinstance(s_grad, torch.Tensor):
                    diff = (
                        s_grad - live_diag[step_i]["grads"][name]
                    ).abs().max().item()
                    max_grad_diff = max(max_grad_diff, diff)
        grad_tol = 0 if step_i == 0 else 5e-4
        assert max_grad_diff <= grad_tol, (
            f"Step {step_i} grads differ: max {max_grad_diff:.2e}"
        )

        # Optimizer: increment step counter
        _step_counter += 1
        new_step = _step_counter
        for si in step_state_indices:
            s_opt_state[si] = s_opt_state[si] + 1

        # Execute inner fn calls
        with torch.no_grad():
            for ci, call in enumerate(inner_fn_calls):
                fn_mod = inner_fn_mods[call["fn_file"]]
                roles = call["arg_roles"]
                n_mut = call["num_mutations"]
                mut_indices = call["mutated_arg_indices"]
                call_order = call["call_order"]
                symint_specs = call["symint_specs"]
                copy_back = call["copy_back_groups"]

                for role in roles:
                    if (role["role"] == "optimizer_attr"
                            and "captured_value" in role):
                        s_opt_state[role["state_idx"]].fill_(
                            role["captured_value"]
                        )
                for role in roles:
                    if (role["role"] == "optimizer_attr"
                            and role.get("is_step_attr")):
                        s_opt_state[role["state_idx"]].fill_(new_step)

                args = []
                for role in roles:
                    r = role["role"]
                    if r == "param":
                        args.append(s_params[role["fw_pos"]])
                    elif r == "grad":
                        args.append(grads[role["fw_pos"]])
                    elif r == "state":
                        args.append(s_opt_state[role["state_idx"]])
                    elif r == "optimizer_attr":
                        args.append(s_opt_state[role["state_idx"]])
                    elif r == "stacked_params":
                        gi = role["group_index"]
                        gp = group_params[gi]
                        args.append(
                            torch.stack([s_params[p] for p in gp])
                        )
                    elif r == "stacked_grads":
                        gi = role["group_index"]
                        gp = group_params[gi]
                        args.append(
                            torch.stack([grads[p] for p in gp])
                        )
                    elif r == "constant":
                        args.append(role["value"])
                    else:
                        args.append(None)

                fx_in = []
                for ca, dim in symint_specs:
                    fx_in.append(args[ca].shape[dim])
                for idx in call_order:
                    fx_in.append(args[idx])

                result_ci = fn_mod.forward(*fx_in)

                if n_mut > 0:
                    r_ci = (
                        result_ci
                        if isinstance(result_ci, (tuple, list))
                        else (result_ci,)
                    )
                    for out_i, arg_i in enumerate(mut_indices):
                        args[arg_i].copy_(r_ci[out_i])

                for gi in copy_back:
                    gp = group_params[gi]
                    for ai_idx, role in enumerate(roles):
                        if (role["role"] == "stacked_params"
                                and role["group_index"] == gi):
                            torch._foreach_copy_(
                                [s_params[p] for p in gp],
                                list(args[ai_idx].unbind(0)),
                            )
                            break

        # Compare params after optimizer
        # bf16 CUDA non-determinism causes small diffs that cascade across steps
        max_post_diff = 0
        for fw_pos, s_tensor in s_params.items():
            name = primal_names[fw_pos]
            if name in live_diag[step_i]["params_after"]:
                diff = (
                    s_tensor - live_diag[step_i]["params_after"][name]
                ).abs().max().item()
                max_post_diff = max(max_post_diff, diff)
        opt_tol = 2e-3 * (step_i + 1)
        assert max_post_diff < opt_tol, (
            f"Step {step_i} post-opt params differ: max {max_post_diff:.2e} (tol {opt_tol:.0e})"
        )
