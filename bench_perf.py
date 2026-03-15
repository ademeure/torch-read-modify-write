#!/usr/bin/env python3
"""Performance benchmark: eager vs capture+replay vs torch.compile vs torch.compile-on-aten.

Validates:
  1. capture+replay (our auto_install system) vs eager
  2. torch.compile on captured aten ops vs torch.compile on original model

Modes:
  eager          - vanilla PyTorch, no compilation
  aten_replay    - auto_install: patch torch.compile → capture aten → replay
  compile        - standard torch.compile(backend="inductor")
  compile_aten   - capture aten → torch.compile the aten fwd/bw → install

Usage:
  .venv/bin/python bench_perf.py --model nanogpt [--warmup 5] [--steps 20]
  .venv/bin/python bench_perf.py --model nanochat [--warmup 5] [--steps 20]
"""

from __future__ import annotations

import argparse
import copy
import gc
import os
import sys
import time
import inspect
import types
from pathlib import Path

_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_root))

import torch
import torch._dynamo

os.environ.setdefault(
    "LIBRARY_PATH",
    "/usr/local/cuda-13.1/targets/x86_64-linux/lib/stubs:"
    + os.environ.get("LIBRARY_PATH", ""),
)


# ── Model builders ──────────────────────────────────────────────────

def _build_nanogpt(device: str) -> dict:
    """NanoGPT: model(idx) → logits, loss = CE(logits, targets)."""
    sys.path.insert(0, str(_root / "test_repo"))
    from model import NanoGPT

    model = NanoGPT(vocab_size=256, block_size=128, n_layer=6, n_head=6, n_embd=192)
    model = model.to(device)
    model.train()

    batch_size, seq_len = 8, 128

    def get_batch(step: int):
        torch.manual_seed(1337 + step)
        idx = torch.randint(0, 256, (batch_size, seq_len), device=device)
        targets = torch.randint(0, 256, (batch_size, seq_len), device=device)
        return idx, targets

    def step_fn(model, batch, optimizer):
        idx, targets = batch
        logits = model(idx)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1)
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()

    return {
        "model": model,
        "get_batch": get_batch,
        "step_fn": step_fn,
        "name": "NanoGPT-6L-192d",
        "model_takes_targets": False,
    }


def _build_nanochat(device: str) -> dict:
    """nanochat: model(idx, targets=targets) → loss."""
    from recipes.nanochat_wrapper import setup
    recipe = setup()
    model = recipe["model"].to(device)
    model.train()

    orig_get_batch = recipe["get_batch"]

    def get_batch(step: int):
        args, kwargs = orig_get_batch(step)
        args = tuple(a.to(device) if isinstance(a, torch.Tensor) else a for a in args)
        kwargs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in kwargs.items()}
        return args, kwargs

    def step_fn(model, batch, optimizer):
        args, kwargs = batch
        loss = model(*args, **kwargs)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()

    return {
        "model": model,
        "get_batch": get_batch,
        "step_fn": step_fn,
        "name": "nanochat-4L",
        "model_takes_targets": True,
    }


# ── Helpers ─────────────────────────────────────────────────────────

def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _reset():
    torch._dynamo.reset()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _time_steps(model, get_batch, optimizer, step_fn, *,
                warmup: int, steps: int, label: str) -> dict:
    losses = []

    for i in range(warmup):
        batch = get_batch(i)
        loss_val = step_fn(model, batch, optimizer)
        if i == 0:
            print(f"  [{label}] warmup step 0: loss={loss_val:.6f}")
    _sync()

    step_times = []
    for i in range(steps):
        batch = get_batch(warmup + i)
        _sync()
        t0 = time.perf_counter()
        loss_val = step_fn(model, batch, optimizer)
        _sync()
        step_times.append(time.perf_counter() - t0)
        losses.append(loss_val)

    avg_ms = sum(step_times) / len(step_times) * 1000
    median_ms = sorted(step_times)[len(step_times) // 2] * 1000
    min_ms = min(step_times) * 1000
    max_ms = max(step_times) * 1000
    mem_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0

    print(f"  [{label}] avg={avg_ms:.2f}ms  median={median_ms:.2f}ms  "
          f"min={min_ms:.2f}ms  max={max_ms:.2f}ms  "
          f"mem={mem_mb:.0f}MB  final_loss={losses[-1]:.6f}")

    return {
        "label": label, "avg_ms": avg_ms, "median_ms": median_ms,
        "min_ms": min_ms, "max_ms": max_ms, "mem_mb": mem_mb,
        "losses": losses, "step_times": step_times,
    }


# ── Mode 1: Eager ──────────────────────────────────────────────────

def bench_eager(recipe: dict, warmup: int, steps: int) -> dict:
    _reset()
    model = copy.deepcopy(recipe["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    return _time_steps(model, recipe["get_batch"], optimizer, recipe["step_fn"],
                       warmup=warmup, steps=steps, label="eager")


# ── Mode 2: auto_install (our capture+replay system) ───────────────

def bench_aten_replay(recipe: dict, warmup: int, steps: int) -> dict:
    """Use auto_install: patch torch.compile → capture on first call → replay."""
    _reset()
    model = copy.deepcopy(recipe["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    import torch_graph.auto_install as ai
    ai.unpatch()
    ai.configure(
        cache_dir=".bench_cache/aten_replay",
        force_recapture=True,
        verbose=True,
        capture_backward=True,
        dynamic=True,
    )
    ai.patch()

    # torch.compile goes through our auto_install
    compiled = torch.compile(model)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    step_fn = recipe["step_fn"]

    def aten_step(_, batch, opt):
        return step_fn(compiled, batch, opt)

    result = _time_steps(compiled, recipe["get_batch"], optimizer, aten_step,
                         warmup=warmup, steps=steps, label="aten_replay")

    ai.unpatch()
    return result


# ── Mode 3: torch.compile (inductor) ──────────────────────────────

def bench_compile(recipe: dict, warmup: int, steps: int) -> dict:
    _reset()
    model = copy.deepcopy(recipe["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    compiled = torch.compile(model, dynamic=True)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    step_fn = recipe["step_fn"]

    def compile_step(_, batch, opt):
        return step_fn(compiled, batch, opt)

    return _time_steps(compiled, recipe["get_batch"], optimizer, compile_step,
                       warmup=warmup, steps=steps, label="compile")


# ── Mode 4: capture aten → torch.compile aten → install ───────────

def bench_compile_aten(recipe: dict, warmup: int, steps: int) -> dict:
    """Capture aten graph, wrap fwd/bw with torch.compile(inductor), install."""
    _reset()
    model = copy.deepcopy(recipe["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    from torch_graph.export import capture_aten_graphs, export_aten_program
    from torch_graph.auto_install import (
        _compute_num_mutations, _find_mutated_buffer_paths, _load_aten_module,
    )
    from torch_graph.install import (
        _parse_param_paths, _detect_symint_slots, _build_slot_specs,
        _assemble_inputs, _normalize_user_inputs, _extract_forward_arg_names,
        _make_buffer_writer,
    )

    # Step 1: capture aten graph
    batch = recipe["get_batch"](0)
    if recipe["model_takes_targets"]:
        capture_args, capture_kwargs = batch
    else:
        capture_args = (batch[0],)
        capture_kwargs = {}

    print(f"  [compile_aten] capturing aten graph...")
    t0 = time.perf_counter()
    output, capture = capture_aten_graphs(
        model, *capture_args,
        run_backward=True,
        dynamic=True,
        **capture_kwargs,
    )
    _sync()
    print(f"  [compile_aten] capture took {(time.perf_counter()-t0)*1000:.0f}ms")

    # Step 2: export + load aten module (for signature/param info)
    cache_dir = Path(".bench_cache/compile_aten")
    cache_dir.mkdir(parents=True, exist_ok=True)
    py_path = str(cache_dir / "bench_aten.py")
    export_aten_program(capture, py_path, include_test_harness=False, skip_pt=True)

    aten_mod = _load_aten_module(Path(py_path))
    param_paths = _parse_param_paths(aten_mod)

    fg_gm = capture.forward_graphs[0].graph_module
    bg_gm = capture.backward_graphs[0].graph_module if capture.backward_graphs else None
    n_mutations = _compute_num_mutations(fg_gm, bg_gm)
    mutated_buffers = _find_mutated_buffer_paths(model, capture.primal_names, n_mutations)

    # Step 3: read original signatures, THEN compile
    orig_forward = aten_mod.forward
    orig_backward = getattr(aten_mod, 'backward', None)

    sig = inspect.signature(orig_forward)
    fw_param_names = list(sig.parameters.keys())
    symint_map = _detect_symint_slots(fw_param_names, orig_forward, param_paths)
    slot_specs, min_user_inputs = _build_slot_specs(model, fw_param_names, param_paths, symint_map)
    buffer_writers = [_make_buffer_writer(model, p) for p in mutated_buffers]

    try:
        orig_fwd_params = _extract_forward_arg_names(model.forward)
    except Exception:
        orig_fwd_params = []

    num_real_outputs = 1

    # Derive expected forward length from backward signature
    expected_fw_len = None
    if orig_backward is not None:
        bw_sig = inspect.signature(orig_backward)
        bw_params = [
            p for p in bw_sig.parameters.values()
            if p.name != "self"
            and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]
        if bw_params:
            n_saved = len(bw_params) - num_real_outputs
            if n_saved >= 0:
                expected_fw_len = n_mutations + num_real_outputs + n_saved

    # NOW compile
    compiled_fw = torch.compile(orig_forward, dynamic=True)
    compiled_bw = torch.compile(orig_backward, dynamic=True) if orig_backward else None

    class _CompiledAtenGraph(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *all_inputs):
            result = compiled_fw(*all_inputs)
            if isinstance(result, tuple):
                mut_vals = result[:n_mutations]
                real_out = result[n_mutations:n_mutations + num_real_outputs]
                saved = result[n_mutations + num_real_outputs:]

                for wb, val in zip(buffer_writers, mut_vals):
                    wb(val)

                tensor_saved, non_tensor_saved = [], {}
                for i, v in enumerate(saved):
                    if isinstance(v, torch.Tensor):
                        tensor_saved.append(v)
                    else:
                        non_tensor_saved[i] = v
                ctx.save_for_backward(*tensor_saved)
                ctx._non_tensor_saved = non_tensor_saved
                ctx._num_saved = len(saved)
                return real_out[0] if num_real_outputs == 1 else real_out
            else:
                ctx._num_saved = 0
                return result

        @staticmethod
        def backward(ctx, *grad_outputs):
            tensors = list(ctx.saved_tensors)
            saved = []
            t_idx = 0
            for i in range(ctx._num_saved):
                if i in ctx._non_tensor_saved:
                    saved.append(ctx._non_tensor_saved[i])
                else:
                    saved.append(tensors[t_idx])
                    t_idx += 1
            if ctx._non_tensor_saved:
                non_tensors = [saved[i] for i in sorted(ctx._non_tensor_saved)]
                tensors_only = [v for v in saved if isinstance(v, torch.Tensor)]
                saved = non_tensors + tensors_only

            bw_result = compiled_bw(*saved, *grad_outputs)
            if not isinstance(bw_result, tuple):
                bw_result = (bw_result,)
            return bw_result

    def compiled_aten_fwd(*args, **kwargs):
        user_inputs = _normalize_user_inputs(args, kwargs, orig_fwd_params)
        all_inputs = _assemble_inputs(user_inputs, slot_specs, min_user_inputs)
        return _CompiledAtenGraph.apply(*all_inputs)

    model.forward = compiled_aten_fwd

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    return _time_steps(model, recipe["get_batch"], optimizer, recipe["step_fn"],
                       warmup=warmup, steps=steps, label="compile_aten")


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Performance benchmark")
    parser.add_argument("--model", choices=["nanogpt", "nanochat"], default="nanogpt")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--modes", nargs="*",
                        default=["eager", "aten_replay", "compile", "compile_aten"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dev_name = torch.cuda.get_device_name(0) if args.device == "cuda" else "CPU"
    print(f"═══ Performance Benchmark ═══")
    print(f"Device: {args.device} ({dev_name})")
    print(f"PyTorch: {torch.__version__}")
    print(f"Warmup: {args.warmup}  Timed steps: {args.steps}")
    print()

    torch.manual_seed(args.seed)
    if args.model == "nanogpt":
        recipe = _build_nanogpt(args.device)
    else:
        recipe = _build_nanochat(args.device)

    n_params = sum(p.numel() for p in recipe["model"].parameters())
    print(f"Model: {recipe['name']}  ({n_params:,} params)")
    print()

    mode_fns = {
        "eager": bench_eager,
        "aten_replay": bench_aten_replay,
        "compile": bench_compile,
        "compile_aten": bench_compile_aten,
    }

    results = {}
    for mode in args.modes:
        print(f"── {mode} ──")
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        results[mode] = mode_fns[mode](recipe, args.warmup, args.steps)
        print()

    # Summary
    print("═══ Summary ═══")
    print(f"{'Mode':<16} {'Median(ms)':>10} {'Avg(ms)':>10} {'Min(ms)':>10} "
          f"{'Max(ms)':>10} {'Mem(MB)':>10} {'vs eager':>10} {'vs compile':>10}")
    print("─" * 96)

    eager_med = results.get("eager", {}).get("median_ms", 1)
    compile_med = results.get("compile", {}).get("median_ms", 1)

    for mode in args.modes:
        r = results[mode]
        ve = f"{r['median_ms']/eager_med:.2f}x" if "eager" in results else "—"
        vc = f"{r['median_ms']/compile_med:.2f}x" if "compile" in results else "—"
        print(f"{r['label']:<16} {r['median_ms']:>10.2f} {r['avg_ms']:>10.2f} "
              f"{r['min_ms']:>10.2f} {r['max_ms']:>10.2f} {r['mem_mb']:>10.0f} "
              f"{ve:>10} {vc:>10}")

    # Loss comparison
    if len(results) > 1:
        print()
        print("═══ Loss Comparison (first 5 timed steps) ═══")
        items = [(m, r) for m, r in results.items() if r["losses"]]
        if len(items) >= 2:
            ref_m, ref_r = items[0]
            for other_m, other_r in items[1:]:
                n = min(5, len(ref_r["losses"]), len(other_r["losses"]))
                max_diff = max(abs(ref_r["losses"][i] - other_r["losses"][i]) for i in range(n))
                print(f"  {ref_m} vs {other_m}: max |loss diff| = {max_diff:.8f}")

    # Key comparisons
    if "compile" in results and "compile_aten" in results:
        print()
        r_c, r_ca = results["compile"], results["compile_aten"]
        ratio = r_ca["median_ms"] / r_c["median_ms"]
        pct = (ratio - 1) * 100
        status = "PASS" if abs(pct) < 15 else "INVESTIGATE"
        print(f"═══ compile vs compile_aten: {ratio:.3f}x ({pct:+.1f}%)  [{status}] ═══")
        if abs(pct) >= 15:
            print("  WARNING: >15% difference — investigate op decomposition")

    if "eager" in results and "aten_replay" in results:
        print()
        r_e, r_a = results["eager"], results["aten_replay"]
        ratio = r_a["median_ms"] / r_e["median_ms"]
        pct = (ratio - 1) * 100
        print(f"═══ eager vs aten_replay: {ratio:.3f}x ({pct:+.1f}%) ═══")
        if ratio > 2.0:
            print("  NOTE: aten replay runs unfused ops — expected ~2-5x slower")


if __name__ == "__main__":
    main()
