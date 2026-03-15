#!/usr/bin/env python3
"""Performance benchmark for autoresearch (FA3 GPT with MuonAdamW).

Modes:
  eager          - vanilla PyTorch (no compile), with bf16 autocast
  aten_replay    - auto_install: patch torch.compile → capture aten → replay
  compile        - torch.compile(model, dynamic=False) — what autoresearch uses
  compile_aten   - capture aten → torch.compile the aten fwd/bw → install

Usage:
  .venv/bin/python bench_autoresearch.py [--warmup 5] [--steps 20] [--depth 4]
"""

from __future__ import annotations

import argparse
import copy
import gc
import inspect
import os
import sys
import time
import types
from pathlib import Path

_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_root))
# autoresearch repo on path for train.py imports (sdpa-blackwell-compat branch)
_ar_root = _root / ".autoresearch_repo"
sys.path.insert(0, str(_ar_root))
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
# Force SDPA on Blackwell (FA3 kernels load but crash at runtime on SM 10.0)
# Patch flash_attention module before train.py imports it
import importlib.util
_fa_spec = importlib.util.spec_from_file_location(
    "flash_attention", str(_ar_root / "flash_attention.py"))
_fa_mod = importlib.util.module_from_spec(_fa_spec)
# Prevent FA3 from loading by temporarily making kernels unimportable
_real_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__
def _block_kernels(name, *args, **kwargs):
    if name == "kernels":
        raise ImportError("blocked for SDPA fallback")
    return _real_import(name, *args, **kwargs)
import builtins
_saved = builtins.__import__
builtins.__import__ = _block_kernels
try:
    _fa_spec.loader.exec_module(_fa_mod)
finally:
    builtins.__import__ = _saved
sys.modules["flash_attention"] = _fa_mod
print(f"Attention backend: {_fa_mod.FLASH_ATTENTION_IMPL}")

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault(
    "LIBRARY_PATH",
    "/usr/local/cuda-13.1/targets/x86_64-linux/lib/stubs:"
    + os.environ.get("LIBRARY_PATH", ""),
)

import torch
import torch._dynamo
torch.set_float32_matmul_precision("high")


# ── Build model from autoresearch code ──────────────────────────────

def _import_autoresearch_classes():
    """Import GPT, GPTConfig, MuonAdamW from autoresearch without running the training loop.

    train.py executes training at module level, so we can't do a normal import.
    Instead, we exec() only the class/function definitions.
    """
    import importlib
    # prepare.py is safe to import normally
    prepare = importlib.import_module("prepare")

    # For train.py, read the source and exec only the parts before the training loop
    train_path = _ar_root / "train.py"
    source = train_path.read_text()

    # Find where the training setup begins (after class definitions)
    # The classes end before "# Hyperparameters" section
    marker = "# ---------------------------------------------------------------------------\n# Hyperparameters"
    idx = source.find(marker)
    if idx < 0:
        raise RuntimeError("Could not find Hyperparameters section in train.py")

    class_source = source[:idx]

    # Build a proper module object so @dataclass can resolve the module
    import types as _types
    train_mod = _types.ModuleType("train")
    train_mod.__file__ = str(train_path)
    sys.modules["train"] = train_mod
    exec(compile(class_source, str(train_path), "exec"), train_mod.__dict__)
    ns = train_mod.__dict__

    return ns["GPT"], ns["GPTConfig"], ns["MuonAdamW"], prepare.Tokenizer


def _build_autoresearch(depth: int = 4, batch_size: int = 8, seq_len: int = 512):
    """Build autoresearch GPT model (smaller config for benchmarking)."""
    GPT, GPTConfig, MuonAdamW, Tokenizer = _import_autoresearch_classes()

    tokenizer = Tokenizer.from_directory()
    vocab_size = tokenizer.get_vocab_size()

    ASPECT_RATIO = 64
    HEAD_DIM = 128
    WINDOW_PATTERN = "SSSL"

    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM

    config = GPTConfig(
        sequence_len=seq_len,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )

    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device="cuda")
    model.init_weights()
    model.train()

    # Build simple batches (random token IDs)
    def get_batch(step: int):
        torch.manual_seed(1337 + step)
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda")
        y = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda")
        return x, y

    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    def step_fn(model, batch, optimizer):
        x, y = batch
        with autocast_ctx:
            loss = model(x, y)
        loss.backward()
        optimizer.step()
        model.zero_grad(set_to_none=True)
        return loss.item()

    return {
        "model": model,
        "get_batch": get_batch,
        "step_fn": step_fn,
        "name": f"autoresearch-{depth}L-{model_dim}d",
        "model_takes_targets": True,
        "compile_kwargs": {"dynamic": False},
    }


# ── Helpers ─────────────────────────────────────────────────────────

def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _reset():
    torch._dynamo.reset()
    gc.collect()
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
    mem_mb = torch.cuda.max_memory_allocated() / (1024**2)

    print(f"  [{label}] avg={avg_ms:.2f}ms  median={median_ms:.2f}ms  "
          f"min={min_ms:.2f}ms  max={max_ms:.2f}ms  "
          f"mem={mem_mb:.0f}MB  final_loss={losses[-1]:.6f}")

    return {
        "label": label, "avg_ms": avg_ms, "median_ms": median_ms,
        "min_ms": min_ms, "max_ms": max_ms, "mem_mb": mem_mb,
        "losses": losses, "step_times": step_times,
    }


def _make_optimizer(model):
    """Build MuonAdamW with autoresearch-style param grouping."""
    return model.setup_optimizer()


# ── Mode 1: Eager ──────────────────────────────────────────────────

def bench_eager(recipe: dict, warmup: int, steps: int) -> dict:
    _reset()
    model = copy.deepcopy(recipe["model"])
    optimizer = _make_optimizer(model)
    torch.cuda.reset_peak_memory_stats()
    return _time_steps(model, recipe["get_batch"], optimizer, recipe["step_fn"],
                       warmup=warmup, steps=steps, label="eager")


# ── Mode 2: auto_install (capture+replay) ──────────────────────────

def bench_aten_replay(recipe: dict, warmup: int, steps: int) -> dict:
    _reset()
    model = copy.deepcopy(recipe["model"])
    optimizer = _make_optimizer(model)

    import torch_graph.auto_install as ai
    ai.unpatch()
    ai.configure(
        cache_dir=".bench_cache/ar_aten_replay",
        force_recapture=True,
        verbose=True,
        capture_backward=True,
        dynamic=False,
    )
    ai.patch()

    compiled = torch.compile(model, **recipe.get("compile_kwargs", {}))

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
    optimizer = _make_optimizer(model)
    compiled = torch.compile(model, **recipe.get("compile_kwargs", {}))

    torch.cuda.reset_peak_memory_stats()

    step_fn = recipe["step_fn"]
    def compile_step(_, batch, opt):
        return step_fn(compiled, batch, opt)

    return _time_steps(compiled, recipe["get_batch"], optimizer, compile_step,
                       warmup=warmup, steps=steps, label="compile")


# ── Mode 4: capture aten → torch.compile aten → install ───────────

def bench_compile_aten(recipe: dict, warmup: int, steps: int) -> dict:
    _reset()
    model = copy.deepcopy(recipe["model"])
    optimizer = _make_optimizer(model)

    from torch_graph.export import capture_aten_graphs, export_aten_program
    from torch_graph.auto_install import (
        _compute_num_mutations, _find_mutated_buffer_paths, _load_aten_module,
    )
    from torch_graph.install import (
        _parse_param_paths, _detect_symint_slots, _build_slot_specs,
        _assemble_inputs, _normalize_user_inputs, _extract_forward_arg_names,
        _make_buffer_writer,
    )

    # Capture with bf16 autocast
    x, y = recipe["get_batch"](0)
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    print(f"  [compile_aten] capturing aten graph...")
    t0 = time.perf_counter()
    with autocast_ctx:
        output, capture = capture_aten_graphs(
            model, x,
            run_backward=True,
            dynamic=False,
            targets=y,
        )
    _sync()
    print(f"  [compile_aten] capture took {(time.perf_counter()-t0)*1000:.0f}ms")

    # Count ops
    fg = capture.forward_graphs[0].graph_module
    from collections import Counter
    ops = Counter(str(n.target) for n in fg.graph.nodes if n.op == 'call_function')
    print(f"  [compile_aten] forward ops: {sum(ops.values())} ({len(ops)} unique)")

    # Export + load
    cache_dir = Path(".bench_cache/ar_compile_aten")
    cache_dir.mkdir(parents=True, exist_ok=True)
    py_path = str(cache_dir / "ar_aten.py")
    export_aten_program(capture, py_path, include_test_harness=False, skip_pt=True)

    aten_mod = _load_aten_module(Path(py_path))
    param_paths = _parse_param_paths(aten_mod)

    bg_gm = capture.backward_graphs[0].graph_module if capture.backward_graphs else None
    n_mutations = _compute_num_mutations(fg, bg_gm)
    mutated_buffers = _find_mutated_buffer_paths(model, capture.primal_names, n_mutations)

    # Read original signatures, THEN compile
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

    # Compile aten forward/backward with inductor.
    # IMPORTANT: use the REAL torch.compile, not our auto_install patched version,
    # otherwise the backward gets intercepted and wrapped in _CompiledFnProxy.
    import torch_graph.auto_install as _ai
    _real_compile = _ai._real_torch_compile or torch.compile
    compiled_fw = _real_compile(orig_forward, dynamic=False)
    compiled_bw = _real_compile(orig_backward, dynamic=False) if orig_backward else None

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

    torch.cuda.reset_peak_memory_stats()

    return _time_steps(model, recipe["get_batch"], optimizer, recipe["step_fn"],
                       warmup=warmup, steps=steps, label="compile_aten")


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Autoresearch benchmark")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--modes", nargs="*",
                        default=["eager", "aten_replay", "compile", "compile_aten"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"═══ Autoresearch Performance Benchmark ═══")
    print(f"Device: cuda ({torch.cuda.get_device_name(0)})")
    print(f"PyTorch: {torch.__version__}")
    print(f"Warmup: {args.warmup}  Timed steps: {args.steps}")
    print(f"Config: depth={args.depth}  batch={args.batch_size}  seq_len={args.seq_len}")
    print()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    recipe = _build_autoresearch(
        depth=args.depth, batch_size=args.batch_size, seq_len=args.seq_len,
    )

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

    if "compile" in results and "compile_aten" in results:
        print()
        r_c, r_ca = results["compile"], results["compile_aten"]
        ratio = r_ca["median_ms"] / r_c["median_ms"]
        pct = (ratio - 1) * 100
        status = "PASS" if abs(pct) < 15 else "INVESTIGATE"
        print(f"═══ compile vs compile_aten: {ratio:.3f}x ({pct:+.1f}%)  [{status}] ═══")

    if "eager" in results and "aten_replay" in results:
        print()
        r_e, r_a = results["eager"], results["aten_replay"]
        ratio = r_a["median_ms"] / r_e["median_ms"]
        pct = (ratio - 1) * 100
        print(f"═══ eager vs aten_replay: {ratio:.3f}x ({pct:+.1f}%) ═══")


if __name__ == "__main__":
    main()
