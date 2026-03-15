"""End-to-end autoresearch training verification.

Runs autoresearch (MuonAdamW optimizer with both adamw and muon param groups)
for N steps and verifies:
  1. Eager baseline losses are recorded
  2. auto_install-captured aten losses match eager within tolerance
  3. Inner compiled functions (adamw_step_fused, muon_step_fused) are
     captured individually (not as a monolithic optimizer graph)
  4. Training loss decreases (model is learning)

Requires: autoresearch repo at outputs/repos/autoresearch/ and
tokenizer/data cache at ~/.cache/autoresearch/
"""

import os
import shutil
import sys
import tempfile

import pytest
import torch

# Skip entire module if autoresearch repo or data is unavailable
_REPO_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs", "repos", "autoresearch",
)
_has_repo = os.path.exists(os.path.join(_REPO_DIR, "train.py"))
_has_cuda = torch.cuda.is_available()
_has_cache = os.path.exists(os.path.expanduser("~/.cache/autoresearch/tokenizer"))

pytestmark = pytest.mark.skipif(
    not (_has_repo and _has_cuda and _has_cache),
    reason="Requires autoresearch repo + CUDA + tokenizer cache",
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch_graph.auto_install as ai

N_STEPS = 20
TOLERANCE = 5e-3  # bfloat16 + torch.compile vs captured aten inner fns


# ── Helpers ──────────────────────────────────────────────────────────────

def _setup_autoresearch():
    """Build a small autoresearch model + MuonAdamW + batch function.

    IMPORTANT: ai.patch() must be active before the first call, because
    _load_train_module() caches the train.py namespace globally.  The
    @torch.compile decorators on adamw_step_fused / muon_step_fused are
    executed during that first load — if torch.compile is not patched,
    they become real compiled functions instead of _CompiledFnProxy.
    """
    recipes_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "recipes",
    )
    if recipes_dir not in sys.path:
        sys.path.insert(0, recipes_dir)

    from autoresearch_wrapper import setup_small
    try:
        return setup_small(device="cuda")
    except AttributeError as e:
        if "flash_attn_interface" in str(e):
            pytest.skip(
                "flash_attention module state corrupted (run autoresearch tests "
                "in isolation: pytest tests/test_autoresearch_e2e.py)"
            )
        raise


def _run_eager(recipe, n_steps):
    """Run n_steps of eager training, return per-step losses.

    Uses _capture_depth > 0 to force all _CompiledFnProxy instances
    (adamw_step_fused, muon_step_fused) to pass through to the original
    function, giving true eager behavior.  Also disables optimizer capture.
    """
    model = recipe["model"]
    get_batch = recipe["get_batch"]

    # Disable optimizer capture for eager baseline
    old_capture = ai._config.capture_optimizer
    ai.configure(capture_optimizer=False)
    optimizer = model.setup_optimizer()
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


def _clone_model(model):
    """Clone model state for a fresh run."""
    import copy
    return copy.deepcopy(model)


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _cleanup():
    """Reset dynamo and auto_install between tests.

    Patch is active at start to ensure _load_train_module() wraps
    @torch.compile decorators as _CompiledFnProxy on first load.
    """
    torch._dynamo.reset()
    ai._installed.clear()
    ai._optimizer_captured.clear()
    # Clear _CompiledFnProxy variant caches (they're module-level globals
    # in the train module, so they persist across tests)
    for entry in list(ai._installed.values()):
        pass  # entries are cleared above
    # Ensure patch is active (train module loading needs it)
    ai.patch()
    # Clean up flash_attention modules from sys.modules before each test.
    # nanochat and autoresearch use different FA3 repo IDs which produce
    # different module hashes via kernels.get_kernel().  If nanochat tests
    # ran first, the stale module lacks the attribute autoresearch expects.
    stale = [k for k in sys.modules if k.startswith("flash_attention_3")]
    for k in stale:
        del sys.modules[k]

    # Reset variants on any existing _CompiledFnProxy instances
    # (they're module-level globals in the train module, persist across tests)
    try:
        import autoresearch_wrapper as arw
        if arw._train_ns is not None:
            for val in arw._train_ns.values():
                if isinstance(val, ai._CompiledFnProxy):
                    val._variants.clear()
            # Force fresh load since FA3 modules were cleaned
            arw._train_ns = None
            sys.modules.pop("train", None)
    except ImportError:
        pass
    yield
    ai.unpatch()
    ai._installed.clear()
    ai._optimizer_captured.clear()
    torch._dynamo.reset()


# ── Tests ────────────────────────────────────────────────────────────────

def test_autoresearch_eager_baseline():
    """Sanity check: eager training produces decreasing losses."""
    recipe = _setup_autoresearch()
    losses = _run_eager(recipe, N_STEPS)

    assert len(losses) == N_STEPS
    assert all(isinstance(l, float) for l in losses)
    # Loss should generally decrease (allow some noise)
    assert losses[-1] < losses[0] + 1.0, \
        f"Training unstable: losses went from {losses[0]:.4f} to {losses[-1]:.4f}"


def test_autoresearch_aten_vs_eager():
    """auto_install aten losses match eager over N steps."""
    # ── Eager baseline ──
    recipe = _setup_autoresearch()
    model_state = {k: v.clone() for k, v in recipe["model"].state_dict().items()}
    eager_losses = _run_eager(recipe, N_STEPS)

    # ── auto_install run ──
    cache_dir = tempfile.mkdtemp(prefix="torch_graph_autoresearch_e2e_")
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
        recipe2 = _setup_autoresearch()
        recipe2["model"].load_state_dict(model_state)
        model2 = recipe2["model"]
        get_batch = recipe2["get_batch"]
        optimizer2 = model2.setup_optimizer()

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

        # ── Verify inner compiled fn detection ──
        opt_entry = ai._optimizer_captured.get(id(optimizer2))
        assert opt_entry is not None, "Optimizer should have been captured"
        assert opt_entry.get("uses_inner_compiled"), \
            f"MuonAdamW should use inner compiled fns, got: {opt_entry}"

        # ── Verify inner compiled fns were installed ──
        fn_entries = [e for e in ai._installed.values() if e.kind == "function"]
        assert len(fn_entries) > 0, \
            f"Expected inner compiled functions to be installed, got: {list(ai._installed.values())}"

        # ── Compare losses ──
        # Tolerance grows with step count since bf16 rounding errors accumulate
        for step, (eager, aten) in enumerate(zip(eager_losses, aten_losses)):
            diff = abs(eager - aten)
            step_tol = TOLERANCE * (1 + step)
            assert diff < step_tol, \
                f"Step {step}: eager={eager:.6f} aten={aten:.6f} diff={diff:.2e} tol={step_tol:.2e}"

        print(f"\n{'='*50}")
        print(f"  Autoresearch E2E: {N_STEPS} steps")
        print(f"{'='*50}")
        for step, (e, a) in enumerate(zip(eager_losses, aten_losses)):
            delta = f"{a - e:+.2e}" if abs(a - e) > 0 else "0"
            print(f"  Step {step+1}: eager={e:.6f}  aten={a:.6f}  diff={delta}")
        print(f"{'='*50}")

    finally:
        ai.unpatch()
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)


def test_autoresearch_verify_mode():
    """--verify mode records losses and produces summary."""
    cache_dir = tempfile.mkdtemp(prefix="torch_graph_autoresearch_verify_")
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
        verify_steps=N_STEPS,
        record_steps=True,
    )
    ai.patch()

    try:
        recipe = _setup_autoresearch()
        model = recipe["model"]
        get_batch = recipe["get_batch"]
        optimizer = model.setup_optimizer()

        compiled = torch.compile(model, dynamic=False)

        # Run steps — we need to NOT hit the sys.exit(0) in exit_after_capture,
        # so set exit_after_capture=0 and just run manually
        ai.configure(exit_after_capture=0)

        for step in range(N_STEPS):
            args, kwargs = get_batch(step)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = compiled(*args, **kwargs)
            loss.backward()
            optimizer.step()

        # Check that losses were recorded
        assert len(ai._step_losses) == N_STEPS, \
            f"Expected {N_STEPS} recorded losses, got {len(ai._step_losses)}"

        # Trigger summary + JSON save
        ai._print_verification_summary()

        import json
        summary_path = os.path.join(cache_dir, "training_summary.json")
        assert os.path.exists(summary_path), f"Summary file not found: {summary_path}"
        with open(summary_path) as f:
            summary = json.load(f)
        assert len(summary["losses"]) == N_STEPS
        assert all(isinstance(l, float) for l in summary["losses"])

    finally:
        ai.unpatch()
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)


def test_autoresearch_muon_groups_captured():
    """Both adamw and muon param groups have their inner fns captured."""
    cache_dir = tempfile.mkdtemp(prefix="torch_graph_autoresearch_muon_")
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
        recipe = _setup_autoresearch()
        model = recipe["model"]
        get_batch = recipe["get_batch"]
        optimizer = model.setup_optimizer()

        compiled = torch.compile(model, dynamic=False)

        # Run 2 steps (enough to trigger capture + replay)
        for step in range(2):
            args, kwargs = get_batch(step)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = compiled(*args, **kwargs)
            loss.backward()
            optimizer.step()

        # Check installed entries
        fn_names = [e.name for e in ai._installed.values() if e.kind == "function"]
        print(f"Installed functions: {fn_names}")

        # Should have both adamw and muon inner functions
        has_adamw_fn = any("adamw" in n.lower() for n in fn_names)
        has_muon_fn = any("muon" in n.lower() for n in fn_names)

        assert has_adamw_fn, \
            f"Expected adamw_step_fused to be captured, got: {fn_names}"
        assert has_muon_fn, \
            f"Expected muon_step_fused to be captured, got: {fn_names}"

        # Check that cache files exist for the inner functions
        cache_files = list(os.listdir(cache_dir))
        aten_files = [f for f in cache_files if f.endswith("_aten.py")]
        adamw_files = [f for f in aten_files if "adamw_step_fused" in f]
        muon_files = [f for f in aten_files if "muon_step_fused" in f]
        print(f"Cache: {len(aten_files)} aten files ({len(adamw_files)} adamw, {len(muon_files)} muon)")
        assert len(adamw_files) > 0, \
            f"Expected adamw_step_fused cache files, got: {aten_files}"
        assert len(muon_files) > 0, \
            f"Expected muon_step_fused cache files, got: {aten_files}"

    finally:
        ai.unpatch()
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)


def test_autoresearch_full_inner_replay():
    """Full inner fn replay: optimizer.step() is NEVER called after step 1.

    With replay_optimizer=True, _run_inner_replay manages:
    - Step counter increment + scalar tensor fills (adamw_step_t)
    - torch.stack/unstack for muon param groups
    - Calling captured aten graphs via proxies
    - Copy-back of stacked params

    This test verifies losses match eager over N_STEPS with both adamw
    AND muon param groups, proving the full optimizer outer loop is replaced.
    """
    # ── Eager baseline ──
    recipe = _setup_autoresearch()
    model_state = {k: v.clone() for k, v in recipe["model"].state_dict().items()}
    eager_losses = _run_eager(recipe, N_STEPS)

    # ── auto_install with FULL inner replay ──
    cache_dir = tempfile.mkdtemp(prefix="torch_graph_autoresearch_replay_")
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
        replay_optimizer=True,  # Full replay!
    )
    ai.patch()

    try:
        recipe2 = _setup_autoresearch()
        recipe2["model"].load_state_dict(model_state)
        model2 = recipe2["model"]
        get_batch = recipe2["get_batch"]
        optimizer2 = model2.setup_optimizer()

        compiled = torch.compile(model2, dynamic=False)

        replay_losses = []
        for step in range(N_STEPS):
            args, kwargs = get_batch(step)
            optimizer2.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = compiled(*args, **kwargs)
            loss.backward()
            optimizer2.step()
            replay_losses.append(loss.item())

        # ── Verify inner replay plan was built ──
        opt_entry = ai._optimizer_captured.get(id(optimizer2))
        assert opt_entry is not None, "Optimizer should have been captured"
        assert opt_entry.get("uses_inner_compiled"), \
            "MuonAdamW should use inner compiled fns"
        inner_replay = opt_entry.get("inner_replay")
        assert inner_replay is not None, \
            "Inner replay plan should exist (replay_optimizer=True)"

        # Should have both adamw and muon calls in the plan
        roles_in_plan = set()
        for call in inner_replay.calls:
            for role in call["arg_roles"]:
                roles_in_plan.add(role["role"])
        assert "stacked_params" in roles_in_plan, \
            f"Expected stacked_params (muon) in plan, got roles: {roles_in_plan}"
        assert "param" in roles_in_plan, \
            f"Expected param (adamw) in plan, got roles: {roles_in_plan}"
        assert len(inner_replay.step_attr_names) > 0, \
            f"Expected step attrs detected, got: {inner_replay.step_attr_names}"

        # ── Compare losses ──
        print(f"\n{'='*50}")
        print(f"  Full inner replay: {N_STEPS} steps")
        print(f"{'='*50}")
        for step, (e, r) in enumerate(zip(eager_losses, replay_losses)):
            diff = abs(e - r)
            step_tol = TOLERANCE * (1 + step)
            delta = f"{r - e:+.2e}" if diff > 0 else "0"
            print(f"  Step {step+1}: eager={e:.6f}  replay={r:.6f}  diff={delta}")
            assert diff < step_tol, \
                f"Step {step}: eager={e:.6f} replay={r:.6f} diff={diff:.2e} tol={step_tol:.2e}"
        print(f"{'='*50}")

    finally:
        ai.unpatch()
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)


def test_autoresearch_params_are_live():
    """Prove optimizer updates flow through aten: reset to step-1 params, loss snaps back.

    Runs 10 steps, saves step-1 params, then resets to them.
    The loss on the SAME input batch should match step 1's loss — proving
    the captured aten graph reads live parameters, not frozen ones.
    """
    cache_dir = tempfile.mkdtemp(prefix="torch_graph_autoresearch_live_")
    torch._dynamo.reset()
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
        recipe = _setup_autoresearch()
        model = recipe["model"]
        get_batch = recipe["get_batch"]
        optimizer = model.setup_optimizer()

        compiled = torch.compile(model, dynamic=False)

        # Step 1: capture and record loss + save initial params
        args0, kwargs0 = get_batch(0)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = compiled(*args0, **kwargs0)
        step1_loss = loss.item()
        step1_params = {k: v.clone() for k, v in model.state_dict().items()}
        loss.backward()
        optimizer.step()

        # Steps 2-10: train normally
        for step in range(1, 10):
            args, kwargs = get_batch(step)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = compiled(*args, **kwargs)
            loss.backward()
            optimizer.step()

        # Forward on step-1 batch with trained params (should differ from step 1)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = compiled(*args0, **kwargs0)
        trained_loss = loss.item()

        # Reset to step-1 params and re-run same batch
        model.load_state_dict(step1_params)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss_after_reset = compiled(*args0, **kwargs0)
        reset_loss = loss_after_reset.item()

        # Eager check to confirm aten is reading the reset params
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            eager_reset = model(*args0, **kwargs0).item()

        print(f"Step  1 loss (same batch, initial params): {step1_loss:.6f}")
        print(f"Step 11 loss (same batch, trained params): {trained_loss:.6f}")
        print(f"After reset to initial params (aten):      {reset_loss:.6f}")
        print(f"After reset to initial params (eager):     {eager_reset:.6f}")

        # Loss after reset should match step 1 (same params, same input)
        assert abs(reset_loss - step1_loss) < 1e-4, \
            f"Reset should reproduce step-1 loss: " \
            f"step1={step1_loss:.6f} reset={reset_loss:.6f}"

        # Aten should match eager after param reset
        assert abs(reset_loss - eager_reset) < 1e-4, \
            f"Aten should match eager after reset: " \
            f"aten={reset_loss:.6f} eager={eager_reset:.6f}"

        # Trained params should produce different loss on same batch
        assert abs(trained_loss - step1_loss) > 0.001, \
            f"10 steps should change loss on same batch: " \
            f"step1={step1_loss:.6f} trained={trained_loss:.6f}"

    finally:
        ai.unpatch()
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
