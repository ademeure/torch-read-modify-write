"""End-to-end nanochat training verification.

Runs nanochat (MuonAdamW optimizer with adamw param groups) for N steps
and verifies:
  1. Eager baseline losses are recorded
  2. auto_install-captured aten losses match eager within tolerance
  3. Inner compiled functions (adamw_step_fused) are captured individually
  4. Live params are verified (sabotage test proves aten reads live weights)

Requires: nanochat repo at outputs/repos/nanochat/ and CUDA GPU (FA3 needs
Hopper/bf16).
"""

import os
import shutil
import sys
import tempfile

import pytest
import torch

# ── Repo / CUDA availability ────────────────────────────────────────────

_REPO_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs", "repos", "nanochat",
)
_has_repo = os.path.exists(os.path.join(_REPO_DIR, "nanochat", "gpt.py"))
_has_cuda = torch.cuda.is_available()

pytestmark = pytest.mark.skipif(
    not (_has_repo and _has_cuda),
    reason="Requires nanochat repo + CUDA GPU",
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch_graph.auto_install as ai

N_STEPS = 5
TOLERANCE = 5e-3  # bfloat16 inner compiled fns vs eager


# ── Helpers ──────────────────────────────────────────────────────────────

def _nanochat_setup():
    """Build a small nanochat GPT model + MuonAdamW + batch generator.

    Returns dict with model, optimizer, get_batch, vocab_size, device.
    Uses the same small config as test_auto_install._nanochat_setup().
    """
    if _REPO_DIR not in sys.path:
        sys.path.insert(0, _REPO_DIR)

    from nanochat.gpt import GPT, GPTConfig
    from nanochat.optim import MuonAdamW

    vocab_size = 50304
    device = torch.device("cuda")

    DEPTH = 2
    HEAD_DIM = 64
    base_dim = DEPTH * 32
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM

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
    model = model.to(device)

    # Build optimizer: all params as adamw groups (same as test_auto_install)
    param_groups = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        param_groups.append({
            "params": [p], "kind": "adamw",
            "lr": 1e-3, "betas": (0.8, 0.95), "eps": 1e-8, "weight_decay": 0.0,
        })
    optimizer = MuonAdamW(param_groups)

    # Deterministic batches keyed by step index
    def get_batch(step):
        g = torch.Generator(device=device)
        g.manual_seed(1000 + step)
        idx = torch.randint(0, vocab_size, (2, 16), device=device, generator=g)
        targets = torch.randint(0, vocab_size, (2, 16), device=device, generator=g)
        return (idx,), {"targets": targets}

    return {
        "model": model,
        "optimizer": optimizer,
        "get_batch": get_batch,
        "vocab_size": vocab_size,
        "device": device,
    }


def _run_eager(recipe, n_steps):
    """Run n_steps of eager training, return per-step losses.

    Uses _capture_depth > 0 to force all _CompiledFnProxy instances
    (adamw_step_fused) to pass through to the original function, giving
    true eager behavior.  Also disables optimizer capture.
    """
    model = recipe["model"]
    optimizer = recipe["optimizer"]
    get_batch = recipe["get_batch"]

    # Disable optimizer capture for eager baseline
    old_capture = ai._config.capture_optimizer
    ai.configure(capture_optimizer=False)
    # Undo the step wrapping that _patched_optimizer_init applied
    if hasattr(optimizer, '_torch_graph_original_step'):
        optimizer.step = optimizer._torch_graph_original_step

    # Set _capture_depth > 0 so inner _CompiledFnProxy pass through
    ai._capture_depth += 1
    losses = []
    try:
        for step in range(n_steps):
            args, kwargs = get_batch(step)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(*args, **kwargs)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    finally:
        ai._capture_depth -= 1
        ai.configure(capture_optimizer=old_capture)

    return losses


def _clone_state(model):
    """Clone model state_dict for later restoration."""
    return {k: v.clone() for k, v in model.state_dict().items()}


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _cleanup():
    """Reset dynamo and auto_install between tests.

    Patch is active at start to ensure @torch.compile decorators on
    adamw_step_fused are intercepted as _CompiledFnProxy on first load.
    """
    torch._dynamo.reset()
    ai._installed.clear()
    ai._optimizer_captured.clear()
    # Ensure patch is active (nanochat optim module loading needs it)
    ai.patch()

    # Clean up flash_attention_3 modules from sys.modules that may have been
    # corrupted by prior tests (e.g. test_auto_install nanochat tests or
    # autoresearch tests with different FA3 repo hashes). Then force-reload
    # the nanochat modules that hold references to FA3 so they re-acquire
    # fresh handles with working fake tensor implementations.
    stale_fa3 = [k for k in sys.modules if k.startswith("flash_attention_3")]
    for k in stale_fa3:
        del sys.modules[k]
    # Force nanochat modules that reference FA3 to reload
    for mod_name in ["nanochat.flash_attention", "nanochat.gpt"]:
        if mod_name in sys.modules:
            del sys.modules[mod_name]

    # Reset _CompiledFnProxy variant caches on nanochat optim globals.
    # These are module-level globals (adamw_step_fused, muon_step_fused)
    # that persist across tests.
    nanochat_optim = sys.modules.get("nanochat.optim")
    if nanochat_optim is not None:
        for name in dir(nanochat_optim):
            val = getattr(nanochat_optim, name, None)
            if isinstance(val, ai._CompiledFnProxy):
                val._variants.clear()

    yield
    ai.unpatch()
    ai._installed.clear()
    ai._optimizer_captured.clear()
    torch._dynamo.reset()


# ── Tests ────────────────────────────────────────────────────────────────

def test_nanochat_aten_vs_eager():
    """auto_install aten losses match eager over N steps."""
    # ── Eager baseline ──
    recipe = _nanochat_setup()
    model_state = _clone_state(recipe["model"])
    eager_losses = _run_eager(recipe, N_STEPS)

    assert len(eager_losses) == N_STEPS
    assert all(isinstance(l, float) for l in eager_losses)

    # ── auto_install run ──
    cache_dir = tempfile.mkdtemp(prefix="torch_graph_nanochat_e2e_")
    torch._dynamo.reset()
    ai.unpatch()
    ai._installed.clear()
    ai._optimizer_captured.clear()

    ai.configure(
        cache_dir=cache_dir,
        verbose=True,
        force_recapture=True,
        dynamic=False,
        capture_backward=True,
        capture_optimizer=True,
        replay_optimizer=False,
    )
    ai.patch()

    try:
        # Fresh model from same initial state
        recipe2 = _nanochat_setup()
        recipe2["model"].load_state_dict(model_state)
        model2 = recipe2["model"]
        optimizer2 = recipe2["optimizer"]
        get_batch = recipe2["get_batch"]

        compiled = torch.compile(model2, dynamic=False)

        aten_losses = []
        for step in range(N_STEPS):
            args, kwargs = get_batch(step)
            optimizer2.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = compiled(*args, **kwargs)
            loss.backward()
            optimizer2.step()
            aten_losses.append(loss.item())

        # ── Compare losses ──
        for step, (eager, aten) in enumerate(zip(eager_losses, aten_losses)):
            diff = abs(eager - aten)
            assert diff < TOLERANCE, \
                f"Step {step}: eager={eager:.6f} aten={aten:.6f} diff={diff:.2e}"

        print(f"\n{'='*50}")
        print(f"  Nanochat E2E: {N_STEPS} steps")
        print(f"{'='*50}")
        for step, (e, a) in enumerate(zip(eager_losses, aten_losses)):
            delta = f"{a - e:+.2e}" if abs(a - e) > 0 else "0"
            print(f"  Step {step+1}: eager={e:.6f}  aten={a:.6f}  diff={delta}")
        print(f"{'='*50}")

    finally:
        ai.unpatch()
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)


def test_nanochat_inner_fns_captured():
    """Verify adamw_step_fused is captured as an inner compiled function."""
    cache_dir = tempfile.mkdtemp(prefix="torch_graph_nanochat_innerfn_")
    torch._dynamo.reset()
    ai.unpatch()
    ai._installed.clear()
    ai._optimizer_captured.clear()

    ai.configure(
        cache_dir=cache_dir,
        verbose=True,
        force_recapture=True,
        dynamic=False,
        capture_backward=True,
        capture_optimizer=True,
    )
    ai.patch()

    try:
        recipe = _nanochat_setup()
        model = recipe["model"]
        optimizer = recipe["optimizer"]
        get_batch = recipe["get_batch"]

        compiled = torch.compile(model, dynamic=False)

        # Run 2 steps (enough to trigger capture + first replay)
        for step in range(2):
            args, kwargs = get_batch(step)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = compiled(*args, **kwargs)
            loss.backward()
            optimizer.step()

        # ── Verify inner compiled fn detection ──
        opt_entry = ai._optimizer_captured.get(id(optimizer))
        assert opt_entry is not None, "Optimizer should have been captured"
        assert opt_entry.get("uses_inner_compiled"), \
            f"MuonAdamW should use inner compiled fns, got: {opt_entry}"

        # ── Verify inner compiled fns were installed ──
        fn_entries = [e for e in ai._installed.values() if e.kind == "function"]
        fn_names = [e.name for e in fn_entries]
        assert len(fn_entries) > 0, \
            f"Expected inner compiled functions to be installed, got: {list(ai._installed.values())}"

        has_adamw_fn = any("adamw" in n.lower() for n in fn_names)
        assert has_adamw_fn, \
            f"Expected adamw_step_fused to be captured, got: {fn_names}"

        # ── Verify cache files exist ──
        cache_files = os.listdir(cache_dir)
        aten_files = [f for f in cache_files if f.endswith("_aten.py")]
        adamw_files = [f for f in aten_files if "adamw_step_fused" in f]
        print(f"Cache: {len(aten_files)} aten files ({len(adamw_files)} adamw)")
        assert len(adamw_files) > 0, \
            f"Expected adamw_step_fused cache files, got: {aten_files}"

        print(f"Installed functions: {fn_names}")

    finally:
        ai.unpatch()
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)


def test_nanochat_live_params_verified():
    """Sabotage params and verify aten matches eager (proves params are live).

    After a few normal steps, randomize all parameters.  The aten graph
    should produce the same loss as the eager model with sabotaged params,
    proving that the captured graph reads live weight data (not frozen
    copies from capture time).
    """
    cache_dir = tempfile.mkdtemp(prefix="torch_graph_nanochat_live_")
    torch._dynamo.reset()
    ai.unpatch()
    ai._installed.clear()
    ai._optimizer_captured.clear()

    ai.configure(
        cache_dir=cache_dir,
        verbose=False,
        force_recapture=True,
        dynamic=False,
        capture_backward=True,
        capture_optimizer=True,
    )
    ai.patch()

    try:
        recipe = _nanochat_setup()
        model = recipe["model"]
        optimizer = recipe["optimizer"]
        get_batch = recipe["get_batch"]

        compiled = torch.compile(model, dynamic=False)

        # Run 3 steps normally to establish captured graphs
        normal_losses = []
        for step in range(3):
            args, kwargs = get_batch(step)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = compiled(*args, **kwargs)
            loss.backward()
            optimizer.step()
            normal_losses.append(loss.item())

        # Save step 4 normal loss for reference
        args, kwargs = get_batch(3)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = compiled(*args, **kwargs)
        normal_step4_loss = loss.item()
        loss.backward()
        optimizer.step()

        # Sabotage: randomize all params
        with torch.no_grad():
            for p in model.parameters():
                p.normal_(0, 5.0)

        # Step 5 with sabotaged params via aten graph
        args5, kwargs5 = get_batch(4)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = compiled(*args5, **kwargs5)
        sabotaged_aten_loss = loss.item()

        # Step 5 with sabotaged params via eager
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            eager_loss = model(*args5, **kwargs5).item()

        # The aten graph MUST see the sabotaged params (match eager)
        diff = abs(sabotaged_aten_loss - eager_loss)
        assert diff < 1e-4, \
            f"Aten should match eager with sabotaged params: " \
            f"aten={sabotaged_aten_loss:.6f} eager={eager_loss:.6f} diff={diff:.2e}"

        # The sabotaged loss should be very different from normal training
        print(f"Normal step 4 loss:            {normal_step4_loss:.4f}")
        print(f"Sabotaged step 5 loss (aten):  {sabotaged_aten_loss:.6f}")
        print(f"Sabotaged step 5 loss (eager): {eager_loss:.6f}")
        print(f"Aten-eager diff:               {diff:.2e}")

    finally:
        ai.unpatch()
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
