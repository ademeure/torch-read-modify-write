#!/usr/bin/env python3
"""
End-to-end validation: autoresearch aten capture vs eager, plus corruption tests.

Tests:
  1. Eager baseline (20 steps)
  2. Aten capture + optimizer replay (20 steps) — must match eager within bf16 tolerance
  3. Corrupt model forward  → loss immediately differs
  4. Corrupt model backward → gradients zeroed, no learning
  5. Corrupt AdamW aten     → weights frozen, no learning
  6. Corrupt Muon aten      → weights frozen, no learning
"""
import os, sys, re, shutil, tempfile
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "recipes"))

import torch_graph.auto_install as ai

N_STEPS      = 20
CORRUPT_STEPS = 5
TOLERANCE    = 5e-3   # per-step bf16 accumulation budget


# ── State reset ────────────────────────────────────────────────────────────

def _reset():
    torch._dynamo.reset()
    ai.unpatch()
    ai._installed.clear()
    ai._optimizer_captured.clear()
    # Clear stale FA3 module handles (nanochat and autoresearch hash differently)
    for k in [k for k in sys.modules if k.startswith("flash_attention_3")]:
        del sys.modules[k]
    # Reset autoresearch module cache so @torch.compile re-wraps correctly
    try:
        import autoresearch_wrapper as arw
        if arw._train_ns is not None:
            for val in arw._train_ns.values():
                if isinstance(val, ai._CompiledFnProxy):
                    val._variants.clear()
            arw._train_ns = None
            sys.modules.pop("train", None)
    except (ImportError, TypeError):
        pass


# ── Recipe setup ───────────────────────────────────────────────────────────

def _setup():
    """Build small autoresearch model + MuonAdamW. ai.patch() must be active."""
    ai.patch()
    from autoresearch_wrapper import setup_small
    try:
        return setup_small(device="cuda")
    except AttributeError as e:
        if "flash_attn_interface" in str(e):
            print("ERROR: flash_attention module state corrupted — run in isolation")
            sys.exit(1)
        raise


# ── Run helpers ────────────────────────────────────────────────────────────

def run_eager(n_steps, get_batch, model_state):
    """n steps of true eager (inner fns pass-through, no capture)."""
    ai.configure(capture_optimizer=False)
    recipe = _setup()
    recipe["model"].load_state_dict(model_state)
    model  = recipe["model"]
    optimizer = model.setup_optimizer()
    if hasattr(optimizer, "_torch_graph_original_step"):
        optimizer.step = optimizer._torch_graph_original_step

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
    return losses


def run_aten(n_steps, get_batch, model_state, cache_dir,
             force_recapture=True, replay_optimizer=True):
    """n steps with aten capture+replay. Returns (losses, list-of-aten-files)."""
    ai.configure(
        cache_dir=cache_dir,
        verbose=False,
        force_recapture=force_recapture,
        dynamic=False,
        capture_backward=True,
        capture_optimizer=True,
        replay_optimizer=replay_optimizer,
        save_json_ir=force_recapture,   # only generate on fresh capture
    )
    recipe = _setup()
    recipe["model"].load_state_dict(model_state)
    model     = recipe["model"]
    optimizer = model.setup_optimizer()
    compiled  = torch.compile(model, dynamic=False)

    losses = []
    for step in range(n_steps):
        args, kwargs = get_batch(step)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = compiled(*args, **kwargs)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    aten_files = sorted(f for f in os.listdir(cache_dir) if f.endswith("_aten.py"))
    return losses, aten_files


# ── Corruption helpers ─────────────────────────────────────────────────────

def _nth_return(content, n):
    """Return the nth 'return (' match object in content."""
    return list(re.finditer(r"^    return \(", content, re.MULTILINE))[n]


def corrupt_forward_double_loss(path):
    """Multiply the first return value of forward() by 2.0 (doubles the loss)."""
    content = open(path).read()
    # The forward function's return is the FIRST 'return (' in the file
    m = _nth_return(content, 0)
    # Extract the first variable name from "return (varname, ..."
    tail    = content[m.end():]
    varname = re.match(r"(\w+)", tail).group(1)
    inject  = f"    {varname} = {varname} * 2.0  # INJECTED: double loss\n"
    patched = content[:m.start()] + inject + content[m.start():]
    open(path, "w").write(patched)
    return varname


def corrupt_backward_zero_grads(path):
    """Zero every gradient tensor in backward()'s return tuple."""
    content = open(path).read()
    # The backward function's return is the LAST 'return (' in the file
    m = _nth_return(content, -1)
    # Extract the full return tuple: 'return (v1, v2, None, v3,)'
    tail_start = m.start()
    line_end   = content.index("\n", tail_start)
    ret_line   = content[tail_start:line_end]
    # Parse variable names, skipping None
    inner      = re.search(r"return \((.+)\)", ret_line).group(1)
    vars_      = [v.strip().rstrip(",") for v in inner.split(",") if v.strip() not in ("", "None")]
    zero_lines = "".join(f"    {v} = {v} * 0.0  # INJECTED: zero grad\n" for v in vars_)
    patched    = content[:tail_start] + zero_lines + content[tail_start:]
    open(path, "w").write(patched)
    return vars_


def corrupt_optimizer_all_outputs(path, scale=1e6):
    """Corrupt ALL return values by multiplying by a large constant.

    Using scale=1e6 makes every output catastrophically wrong regardless of
    which output position corresponds to which tensor role.
    """
    content  = open(path).read()

    # Parse all return variable names
    m         = _nth_return(content, 0)
    tail      = content[m.end():]
    ret_line  = tail.split("\n")[0]   # "var1, var2, var3,)"
    ret_vars  = [v.strip().rstrip(",)") for v in ret_line.split(",") if v.strip().rstrip(",)")]

    inject  = "".join(
        f"    {v} = {v} * {scale}  # INJECTED: catastrophic corruption\n"
        for v in ret_vars
    )
    patched = content[:m.start()] + inject + content[m.start():]
    open(path, "w").write(patched)
    return ret_vars


# ── Comparison helpers ─────────────────────────────────────────────────────

def compare_losses(label, losses_a, losses_b, name_a="eager", name_b="aten",
                   tolerance=None, expect_match=True):
    """Print a side-by-side table and return pass/fail."""
    tol = tolerance or TOLERANCE
    diffs = [abs(a - b) for a, b in zip(losses_a, losses_b)]
    max_diff = max(diffs)
    per_step_tol = [tol * (1 + i) for i in range(len(losses_a))]
    all_within = all(d < t for d, t in zip(diffs, per_step_tol))

    if expect_match:
        status = "PASS ✓" if all_within else "FAIL ✗"
    else:
        # We WANT them to differ — pass if max_diff is large enough
        status = "PASS ✓ (diverged as expected)" if max_diff > 0.05 else "FAIL ✗ (should have diverged)"

    print(f"\n{'─'*62}")
    print(f"  {label}  [{status}]")
    print(f"{'─'*62}")
    print(f"  {'step':>4}  {name_a:>12}  {name_b:>12}  {'diff':>10}")
    print(f"  {'────':>4}  {'────────────':>12}  {'────────────':>12}  {'──────────':>10}")
    for i, (a, b, d) in enumerate(zip(losses_a, losses_b, diffs)):
        flag = " ←" if (expect_match and d >= per_step_tol[i]) or (not expect_match and i == 0 and d > 0.05) else ""
        print(f"  {i+1:>4}  {a:>12.6f}  {b:>12.6f}  {d:>10.2e}{flag}")
    print(f"{'─'*62}")
    print(f"  max diff: {max_diff:.2e}")
    return "PASS" in status


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    if not torch.cuda.is_available():
        print("CUDA required"); sys.exit(1)

    # ── Bootstrap: capture fresh into a dedicated dir ──────────────────────
    _reset()
    ai.patch()
    initial = _setup()
    model_state = {k: v.clone() for k, v in initial["model"].state_dict().items()}
    get_batch   = initial["get_batch"]

    cache_dir = tempfile.mkdtemp(prefix="torch_graph_validate_autoresearch_")
    print(f"\nCache dir: {cache_dir}")

    results = {}

    try:
        # ── Test 1: Eager baseline ─────────────────────────────────────────
        print("\n" + "="*62)
        print("  TEST 1: Eager baseline")
        print("="*62)
        _reset()
        eager_losses = run_eager(N_STEPS, get_batch, model_state)
        print(f"  Eager losses: {[f'{l:.4f}' for l in eager_losses]}")

        # ── Test 2: Aten capture + replay ──────────────────────────────────
        print("\n" + "="*62)
        print("  TEST 2: Aten capture + replay (20 steps)")
        print("="*62)
        _reset()
        aten_losses, aten_files = run_aten(
            N_STEPS, get_batch, model_state, cache_dir,
            force_recapture=True, replay_optimizer=True,
        )

        json_files = sorted(f for f in os.listdir(cache_dir) if f.endswith("_aten.json"))
        print(f"\n  Captured {len(aten_files)} aten files  ({len(json_files)} JSON IR files):")
        for f in aten_files:
            has_json = f.replace(".py", ".json") in json_files
            print(f"    {f}{'  [+JSON IR]' if has_json else ''}")
        if json_files:
            sample = json_files[0]
            import json as _json
            ir = _json.load(open(os.path.join(cache_dir, sample)))
            schema = ir.get("schema", "?")
            # Count nodes across forward/backward/optimizer sections
            def _count_nodes(obj):
                if isinstance(obj, dict) and "nodes" in obj:
                    return len(obj["nodes"])
                return 0
            n_nodes = sum(_count_nodes(ir.get(k, {})) for k in ("forward", "backward", "optimizer"))
            fw_nodes = _count_nodes(ir.get("forward", {}))
            print(f"\n  JSON IR sample ({sample}):")
            print(f"    schema:   {schema}")
            print(f"    sections: {[k for k in ('forward','backward','optimizer') if k in ir]}")
            print(f"    fw nodes: {fw_nodes}  total: {n_nodes}")

        results["aten_vs_eager"] = compare_losses(
            "Aten vs Eager (20 steps)", eager_losses, aten_losses,
            name_a="eager", name_b="aten", expect_match=True,
        )

        # Identify files by role
        model_files  = [f for f in aten_files if f.lower().startswith("gpt")]
        adamw_files  = [f for f in aten_files if "adamw_step_fused" in f]
        muon_files   = [f for f in aten_files if "muon_step_fused"  in f]

        print(f"\n  File roles:")
        print(f"    model  ({len(model_files)}): {model_files}")
        print(f"    adamw  ({len(adamw_files)}): {adamw_files[:2]}{'...' if len(adamw_files)>2 else ''}")
        print(f"    muon   ({len(muon_files )}): {muon_files[:2]}{'...' if len(muon_files)>2 else ''}")

        # ── Corruption test helper ─────────────────────────────────────────
        def run_corruption_test(label, corrupt_fn, files_to_corrupt,
                                n_steps=CORRUPT_STEPS):
            """Apply corruption, run, compare against both eager and ref aten."""
            backups = {}
            for f in files_to_corrupt:
                p = os.path.join(cache_dir, f)
                backups[p] = open(p).read()
                corrupt_fn(p)

            _reset()
            try:
                corrupt_losses, _ = run_aten(
                    n_steps, get_batch, model_state, cache_dir,
                    force_recapture=False, replay_optimizer=True,
                )
            except Exception as e:
                # NaN/inf from catastrophic corruption is expected
                print(f"  (run crashed with: {e!r})")
                corrupt_losses = [float("nan")] * n_steps
            finally:
                for p, orig in backups.items():
                    open(p, "w").write(orig)

            # Compare vs eager
            p1 = compare_losses(
                f"{label} — vs eager",
                eager_losses[:n_steps], corrupt_losses,
                name_a="eager", name_b="corrupt",
                expect_match=False,
            )
            # Also compare vs reference aten (extra sanity check)
            p2 = compare_losses(
                f"{label} — vs ref aten",
                aten_losses[:n_steps], corrupt_losses,
                name_a="aten", name_b="corrupt",
                expect_match=False,
            )
            return p1 or p2

        # ── Test 3: Corrupt model forward ──────────────────────────────────
        print("\n" + "="*62)
        print("  TEST 3: Corrupt model FORWARD (double the loss output)")
        print("="*62)
        if model_files:
            results["corrupt_fw"] = run_corruption_test(
                "Corrupted forward vs Eager (5 steps)",
                corrupt_forward_double_loss,
                model_files,
            )
        else:
            print("  SKIP: no model aten files found"); results["corrupt_fw"] = None

        # ── Test 4: Corrupt model backward ─────────────────────────────────
        print("\n" + "="*62)
        print("  TEST 4: Corrupt model BACKWARD (zero all gradients)")
        print("="*62)
        if model_files:
            results["corrupt_bw"] = run_corruption_test(
                "Corrupted backward vs Eager (5 steps)",
                corrupt_backward_zero_grads,
                model_files,
            )
        else:
            print("  SKIP: no model aten files found"); results["corrupt_bw"] = None

        # ── Test 5: Corrupt AdamW optimizer ────────────────────────────────
        print("\n" + "="*62)
        print("  TEST 5: Corrupt AdamW aten (all outputs × 1e6)")
        print("="*62)
        if adamw_files:
            results["corrupt_adamw"] = run_corruption_test(
                "Corrupted AdamW",
                corrupt_optimizer_all_outputs,
                adamw_files,
            )
        else:
            print("  SKIP: no adamw aten files found"); results["corrupt_adamw"] = None

        # ── Test 6: Corrupt Muon optimizer ─────────────────────────────────
        print("\n" + "="*62)
        print("  TEST 6: Corrupt Muon aten (all outputs × 1e6)")
        print("="*62)
        if muon_files:
            results["corrupt_muon"] = run_corruption_test(
                "Corrupted Muon",
                corrupt_optimizer_all_outputs,
                muon_files,
            )
        else:
            print("  SKIP: no muon aten files found"); results["corrupt_muon"] = None

        # ── Summary ────────────────────────────────────────────────────────
        print("\n" + "="*62)
        print("  SUMMARY")
        print("="*62)
        all_pass = True
        for name, passed in results.items():
            if passed is None:
                print(f"  {name:30s}  SKIP")
            else:
                flag = "PASS ✓" if passed else "FAIL ✗"
                print(f"  {name:30s}  {flag}")
                all_pass = all_pass and passed
        print("="*62)
        print(f"  Overall: {'ALL PASS ✓' if all_pass else 'SOME FAILURES ✗'}")
        print("="*62)

    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)
        _reset()


if __name__ == "__main__":
    main()
