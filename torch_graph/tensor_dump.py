"""Tensor dumping from real workloads and comparison with exported aten programs.

Captures all intermediate tensors from a real model execution, then runs the
exported aten-level forward/backward functions with only the initial inputs
and compares every intermediate against the reference.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.fx import GraphModule
from torch.fx.interpreter import Interpreter

from torch_graph._utils import is_fake, materialize_tensor as _materialize_tensor, RecordingInterpreter as _RecordingInterpreter, _target_device

logger = logging.getLogger(__name__)


@dataclass
class TensorComparison:
    """Result of comparing a single tensor."""

    name: str
    matches: bool
    max_abs_diff: float = 0.0
    mean_abs_diff: float = 0.0
    max_rel_diff: float = 0.0
    ref_shape: list[int] = field(default_factory=list)
    actual_shape: list[int] = field(default_factory=list)
    ref_dtype: str = ""
    actual_dtype: str = ""
    shape_mismatch: bool = False
    missing: bool = False

    def __repr__(self) -> str:
        if self.missing:
            return f"TensorComparison({self.name}, MISSING)"
        if self.shape_mismatch:
            return f"TensorComparison({self.name}, SHAPE MISMATCH: {self.ref_shape} vs {self.actual_shape})"
        status = "OK" if self.matches else "MISMATCH"
        return (
            f"TensorComparison({self.name}, {status}, "
            f"max_abs={self.max_abs_diff:.2e}, max_rel={self.max_rel_diff:.2e})"
        )


@dataclass
class DumpResult:
    """Result of dumping and comparing tensors."""

    reference_tensors: dict[str, torch.Tensor]
    actual_tensors: dict[str, torch.Tensor]
    comparisons: list[TensorComparison]
    kind: str = "forward"

    @property
    def all_match(self) -> bool:
        return all(c.matches for c in self.comparisons)

    @property
    def num_mismatches(self) -> int:
        return sum(1 for c in self.comparisons if not c.matches)

    def oneline(self) -> str:
        total = len(self.comparisons)
        ok = sum(1 for c in self.comparisons if c.matches)
        sym = "pass" if self.all_match else "FAIL"
        return f"{self.kind}: {ok}/{total} {sym}"

    def summary(self) -> str:
        total = len(self.comparisons)
        ok = sum(1 for c in self.comparisons if c.matches)
        fail = total - ok
        lines = [f"Tensor comparison ({self.kind}): {ok}/{total} passed, {fail} failed"]
        for c in self.comparisons:
            if not c.matches:
                lines.append(f"  FAIL: {c}")
        if self.all_match:
            lines.append("  All tensors match!")
        return "\n".join(lines)

    def report(self) -> str:
        buf = StringIO()
        buf.write(f"{'Name':<30} {'Status':<10} {'MaxAbs':>12} {'MeanAbs':>12} {'MaxRel':>12} {'Shape'}\n")
        buf.write("-" * 100 + "\n")
        for c in self.comparisons:
            status = "OK" if c.matches else "FAIL"
            if c.missing:
                status = "MISSING"
            elif c.shape_mismatch:
                status = "SHAPE"
            buf.write(
                f"{c.name:<30} {status:<10} "
                f"{c.max_abs_diff:>12.2e} {c.mean_abs_diff:>12.2e} {c.max_rel_diff:>12.2e} "
                f"{c.ref_shape}\n"
            )
        return buf.getvalue()


def _materialize_inputs(inputs: list[torch.Tensor]) -> list[torch.Tensor]:
    """Ensure all inputs are real (non-Fake) tensors."""
    return [_materialize_tensor(t) if isinstance(t, torch.Tensor) else t for t in inputs]


def trace_all_intermediates(
    graph_module: GraphModule,
    *args,
) -> dict[str, torch.Tensor]:
    """Run a GraphModule node-by-node and record every intermediate tensor."""
    real_args = _materialize_inputs(list(args))
    interp = _RecordingInterpreter(graph_module)
    interp.run(*real_args)
    return interp.recorded


def _run_forward(
    graph_module: GraphModule,
    *args,
) -> tuple[dict[str, torch.Tensor], Any]:
    """Run forward graph, returning (intermediates, raw_output_tuple).

    Raw output includes saved tensors needed to construct backward inputs.
    """
    real_args = _materialize_inputs(list(args))
    interp = _RecordingInterpreter(graph_module)
    interp.run(*real_args)
    return interp.recorded, interp.final_output


def compare_tensors(
    reference: dict[str, torch.Tensor],
    actual: dict[str, torch.Tensor],
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> list[TensorComparison]:
    """Compare two sets of named tensors element-wise."""
    comparisons = []

    for name, ref in sorted(reference.items()):
        if name not in actual:
            comparisons.append(TensorComparison(
                name=name, matches=False, missing=True,
                ref_shape=list(ref.shape), ref_dtype=str(ref.dtype),
            ))
            continue

        act = actual[name]

        if list(ref.shape) != list(act.shape):
            comparisons.append(TensorComparison(
                name=name, matches=False, shape_mismatch=True,
                ref_shape=list(ref.shape), actual_shape=list(act.shape),
                ref_dtype=str(ref.dtype), actual_dtype=str(act.dtype),
            ))
            continue

        ref_f = ref.float().cpu()
        act_f = act.float().cpu()

        if ref_f.numel() == 0:
            comparisons.append(TensorComparison(
                name=name, matches=True,
                ref_shape=list(ref.shape), actual_shape=list(act.shape),
                ref_dtype=str(ref.dtype), actual_dtype=str(act.dtype),
            ))
            continue

        abs_diff = (ref_f - act_f).abs()
        max_abs = abs_diff.max().item()
        mean_abs = abs_diff.mean().item()

        denom = ref_f.abs().clamp(min=1e-12)
        rel_diff = abs_diff / denom
        max_rel = rel_diff.max().item()

        matches = torch.allclose(ref_f, act_f, atol=atol, rtol=rtol)

        comparisons.append(TensorComparison(
            name=name,
            matches=matches,
            max_abs_diff=max_abs,
            mean_abs_diff=mean_abs,
            max_rel_diff=max_rel,
            ref_shape=list(ref.shape),
            actual_shape=list(act.shape),
            ref_dtype=str(ref.dtype),
            actual_dtype=str(act.dtype),
        ))

    return comparisons


def compute_tensor_stats(tensors: dict[str, torch.Tensor]) -> dict[str, dict]:
    """Compute per-tensor statistics, reducing on-device to avoid slow GPU->CPU transfers."""
    import math
    stats = {}
    for name, t in tensors.items():
        if not isinstance(t, torch.Tensor) or t.numel() == 0:
            continue
        is_float = t.is_floating_point()
        tf = t if t.dtype in (torch.float32, torch.float64) else t.float()
        n = t.numel()

        if n > 1:
            t_var, t_mean = torch.var_mean(tf)
        else:
            t_var = tf.new_zeros(())
            t_mean = tf.reshape(())

        t_min, t_max = torch.aminmax(tf)
        t_abs_mean = tf.abs().mean()
        t_n_zero = (tf == 0).sum()

        if is_float:
            t_has_nan = tf.isnan().any()
            t_has_inf = tf.isinf().any()

        if t.is_cuda:
            # GPU: batch all scalars into one transfer (avoids N sync roundtrips)
            if is_float:
                parts = [t_mean, t_var, t_min, t_max, t_abs_mean, t_n_zero.float(),
                         t_has_nan.float(), t_has_inf.float()]
            else:
                parts = [t_mean, t_var, t_min, t_max, t_abs_mean, t_n_zero.float()]
            vals = torch.stack(parts).cpu().tolist()
            s_mean, s_var, s_min, s_max, s_abs, n_zero = vals[:6]
            has_nan = bool(vals[6]) if is_float else False
            has_inf = bool(vals[7]) if is_float else False
        else:
            # CPU: direct .item() is faster than torch.stack allocation
            s_mean = t_mean.item()
            s_var = t_var.item()
            s_min = t_min.item()
            s_max = t_max.item()
            s_abs = t_abs_mean.item()
            n_zero = t_n_zero.item()
            has_nan = bool(t_has_nan) if is_float else False
            has_inf = bool(t_has_inf) if is_float else False

        stats[name] = {
            "numel": n, "shape": list(t.shape), "dtype": str(t.dtype),
            "mean": s_mean, "std": math.sqrt(s_var),
            "min": s_min, "max": s_max,
            "pct_zero": (n_zero / n) * 100,
            "has_nan": has_nan, "has_inf": has_inf, "abs_mean": s_abs,
        }
    return stats


def _placeholder_has_symbolic_shape(ph) -> bool:
    """Return True if placeholder's val.shape contains symbolic (SymInt) dimensions."""
    val = ph.meta.get("val")
    if val is None or not hasattr(val, "shape"):
        return False
    try:
        from torch.fx.experimental.symbolic_shapes import is_concrete_int
        for dim in val.shape:
            if not is_concrete_int(dim):
                return True
    except ImportError:
        if hasattr(torch, "SymInt"):
            for dim in val.shape:
                if isinstance(dim, torch.SymInt):
                    return True
    return False


def _build_real_forward_inputs(
    model: nn.Module,
    user_inputs: tuple,
    graph_module: GraphModule,
    example_inputs: list[torch.Tensor] | None = None,
) -> list[torch.Tensor]:
    """Reproduce aot_autograd's input ordering: [*flat_params, *flat_buffers, *user_inputs].

    Falls back to shape-matching when ordering doesn't line up (e.g. dynamic shapes
    add extra SymInt placeholders).
    """
    placeholders = [n for n in graph_module.graph.nodes if n.op == "placeholder"]

    params_dict = dict(model.named_parameters(remove_duplicate=False))
    buffers_dict = dict(model.named_buffers(remove_duplicate=False))

    ordered_values = list(params_dict.values()) + list(buffers_dict.values())

    real_inputs = [p.detach().clone() for p in ordered_values]
    for a in user_inputs:
        if isinstance(a, torch.Tensor):
            real_inputs.append(a.detach().clone())
        else:
            real_inputs.append(a)

    has_symbolic = any(_placeholder_has_symbolic_shape(ph) for ph in placeholders)
    has_symint_placeholder = any(
        ph.meta.get("val") is not None and not hasattr(ph.meta["val"], "shape")
        for ph in placeholders
    )

    if len(real_inputs) != len(placeholders):
        if has_symbolic or has_symint_placeholder:
            # Dynamic shapes: graph may have extra SymInt placeholders.
            # Use example_inputs from capture if available and length matches.
            if example_inputs is not None and len(example_inputs) == len(placeholders):
                result = []
                user_tensors = [a for a in user_inputs if isinstance(a, torch.Tensor)]
                for i, (ex, ph) in enumerate(zip(example_inputs, placeholders)):
                    val = ph.meta.get("val")
                    if val is not None and not hasattr(val, "shape"):
                        # SymInt placeholder: use concrete value from user input
                        try:
                            from torch.fx.experimental.symbolic_shapes import is_concrete_int
                            if is_concrete_int(val):
                                result.append(int(val))
                            elif user_tensors:
                                result.append(int(user_tensors[0].shape[0]))
                            else:
                                result.append(2)
                        except Exception as e:
                            logger.debug("Symbolic shape detection failed, defaulting to batch size 2: %s", e)
                            result.append(2)
                    elif isinstance(ex, torch.Tensor):
                        try:
                            result.append(_materialize_tensor(ex))
                        except Exception:
                            # FakeTensor with symbolic shape: use matching user input
                            if val is not None and hasattr(val, "shape") and user_tensors:
                                ph_shape = val.shape
                                for ut in user_tensors:
                                    if len(ut.shape) == len(ph_shape):
                                        result.append(ut.detach().clone())
                                        break
                                else:
                                    raise
                            else:
                                raise
                    else:
                        result.append(ex)
                return result
        return _build_forward_inputs_by_shape(model, user_inputs, graph_module)

    if has_symbolic:
        return real_inputs

    for inp, ph in zip(real_inputs, placeholders):
        val = ph.meta.get("val")
        if val is not None and hasattr(val, "shape") and isinstance(inp, torch.Tensor):
            if tuple(inp.shape) != tuple(val.shape):
                return _build_forward_inputs_by_shape(model, user_inputs, graph_module)

    return real_inputs


def _build_forward_inputs_by_shape(
    model: nn.Module,
    user_inputs: tuple,
    graph_module: GraphModule,
) -> list[torch.Tensor]:
    """Fallback: match graph placeholders to model params by shape."""
    placeholders = [n for n in graph_module.graph.nodes if n.op == "placeholder"]

    param_pool = [(name, p.detach().clone()) for name, p in model.named_parameters()]
    buffer_pool = [(name, b.detach().clone()) for name, b in model.named_buffers()]
    user_pool = [
        a.detach().clone() if isinstance(a, torch.Tensor) else a
        for a in user_inputs
    ]

    param_used = [False] * len(param_pool)
    buffer_used = [False] * len(buffer_pool)
    user_idx = 0

    # Infer a fallback device from model parameters
    _fallback_dev = torch.device("cpu")
    if param_pool:
        _fallback_dev = param_pool[0][1].device

    real_inputs = []
    for ph in placeholders:
        val = ph.meta.get("val")
        if val is None or not hasattr(val, "shape"):
            if user_idx < len(user_pool) and isinstance(user_pool[user_idx], torch.Tensor):
                real_inputs.append(user_pool[user_idx])
                user_idx += 1
            else:
                real_inputs.append(torch.randn(1, device=_fallback_dev))
            continue

        target_shape = tuple(val.shape)
        target_dtype = val.dtype
        target_device = _target_device(val)
        matched = False

        for i, (name, tensor) in enumerate(param_pool):
            if not param_used[i] and tuple(tensor.shape) == target_shape and tensor.dtype == target_dtype:
                real_inputs.append(tensor.to(target_device))
                param_used[i] = True
                matched = True
                break

        if not matched:
            for i, (name, tensor) in enumerate(buffer_pool):
                if not buffer_used[i] and tuple(tensor.shape) == target_shape and tensor.dtype == target_dtype:
                    real_inputs.append(tensor.to(target_device))
                    buffer_used[i] = True
                    matched = True
                    break

        if not matched:
            if user_idx < len(user_pool) and isinstance(user_pool[user_idx], torch.Tensor):
                real_inputs.append(user_pool[user_idx].to(target_device))
                user_idx += 1
            else:
                shape = list(target_shape)
                if target_dtype.is_floating_point:
                    real_inputs.append(torch.randn(shape, dtype=target_dtype, device=target_device))
                else:
                    real_inputs.append(torch.zeros(shape, dtype=target_dtype, device=target_device))

    return real_inputs


def _build_backward_inputs_from_forward(
    forward_output: Any,
    model_output: Any,
    bw_graph_module: GraphModule,
    bw_example_inputs: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Build backward inputs from real forward outputs.

    Forward returns: (mutated_inputs..., user_outputs..., saved_tensors...)
    Backward takes:  (saved_tensors..., tangents_for_user_outputs...)
    """
    if isinstance(forward_output, (tuple, list)):
        fw_flat = [x for x in forward_output]
    else:
        fw_flat = [forward_output]

    if isinstance(model_output, torch.Tensor):
        n_user = 1
    elif isinstance(model_output, (tuple, list)):
        n_user = sum(1 for x in model_output if isinstance(x, torch.Tensor))
    else:
        n_user = 1

    bw_phs = [n for n in bw_graph_module.graph.nodes if n.op == "placeholder"]
    n_bw = len(bw_phs)

    # n_bw = n_saved + n_tangents; one tangent per user output tensor
    n_tangents = n_user
    n_saved = n_bw - n_tangents

    # Mutated inputs precede user outputs in the forward output tuple
    n_total_fw = len(fw_flat)
    n_mutated = n_total_fw - n_user - max(n_saved, 0)
    if n_mutated < 0:
        n_mutated = 0

    saved_start = n_mutated + n_user
    saved_end = saved_start + n_saved

    real_inputs = []

    # Infer target device from forward outputs
    _bw_device = torch.device("cpu")
    for x in fw_flat:
        if isinstance(x, torch.Tensor) and x.device.type != "meta":
            _bw_device = x.device
            break

    # Extract saved tensors from forward output
    for i in range(n_saved):
        fw_idx = saved_start + i
        if fw_idx < len(fw_flat) and isinstance(fw_flat[fw_idx], torch.Tensor):
            real_inputs.append(fw_flat[fw_idx].detach().clone())
        elif i < len(bw_example_inputs):
            real_inputs.append(_materialize_tensor(bw_example_inputs[i]))
        else:
            real_inputs.append(torch.randn(1, device=_bw_device))

    # Tangent tensors (ones): gradient seeds for each user output
    for i in range(n_tangents):
        fw_idx = n_mutated + i
        if fw_idx < len(fw_flat) and isinstance(fw_flat[fw_idx], torch.Tensor):
            real_inputs.append(torch.ones_like(fw_flat[fw_idx].detach()))
        elif (n_saved + i) < len(bw_example_inputs):
            ex = bw_example_inputs[n_saved + i]
            shape = list(ex.shape)
            dtype = ex.dtype
            device = _target_device(ex)
            if dtype.is_floating_point:
                real_inputs.append(torch.ones(shape, dtype=dtype, device=device))
            else:
                real_inputs.append(torch.zeros(shape, dtype=dtype, device=device))
        else:
            real_inputs.append(torch.randn(1, device=_bw_device))

    # Validate shapes and fall back to shape-matching if mismatch
    valid = len(real_inputs) == n_bw
    if valid:
        for inp, ph in zip(real_inputs, bw_phs):
            val = ph.meta.get("val")
            if val is not None and hasattr(val, "shape") and isinstance(inp, torch.Tensor):
                if tuple(inp.shape) != tuple(val.shape):
                    valid = False
                    break

    if not valid:
        return _build_backward_inputs_fallback(
            fw_flat, bw_graph_module, bw_example_inputs
        )

    return real_inputs


def _build_backward_inputs_fallback(
    fw_outputs: list,
    bw_graph_module: GraphModule,
    bw_example_inputs: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Fallback: match backward placeholders to forward outputs by shape."""
    bw_phs = [n for n in bw_graph_module.graph.nodes if n.op == "placeholder"]
    fw_tensors = [x for x in fw_outputs if isinstance(x, torch.Tensor)]
    fw_used = [False] * len(fw_tensors)

    _fallback_device = torch.device("cpu")
    for ft in fw_tensors:
        if ft.device.type != "meta":
            _fallback_device = ft.device
            break

    real_inputs = []
    for i, ph in enumerate(bw_phs):
        if i < len(bw_example_inputs):
            ex = bw_example_inputs[i]
        else:
            val = ph.meta.get("val")
            ex = val if val is not None else None

        if ex is None or not hasattr(ex, "shape"):
            real_inputs.append(torch.randn(1, device=_fallback_device))
            continue

        target_shape = tuple(ex.shape)
        target_dtype = ex.dtype
        target_device = _target_device(ex)

        matched = False
        for fi in range(len(fw_tensors)):
            if fw_used[fi]:
                continue
            ft = fw_tensors[fi]
            if tuple(ft.shape) == target_shape and ft.dtype == target_dtype:
                real_inputs.append(ft.detach().clone())
                fw_used[fi] = True
                matched = True
                break

        if not matched:
            if target_dtype.is_floating_point:
                real_inputs.append(torch.ones(list(target_shape), dtype=target_dtype, device=target_device))
            else:
                real_inputs.append(torch.zeros(list(target_shape), dtype=target_dtype, device=target_device))

    return real_inputs


def dump_and_compare(
    model: nn.Module,
    *args,
    output_dir: str = ".",
    atol: float = 1e-5,
    rtol: float = 1e-4,
    run_backward: bool = True,
    loss_fn: Callable | None = None,
    save_tensors: bool = False,
    verbose: bool = True,
    **kwargs,
) -> list[DumpResult]:
    """Capture aten graphs, run with real tensors twice, verify reproducibility."""
    from torch_graph.export import capture_aten_graphs

    # Kwargs for capture (not passed to model.forward)
    _CAPTURE_ONLY = {"dynamic", "record_real_tensors"}
    capture_only = {k: kwargs[k] for k in _CAPTURE_ONLY if k in kwargs}
    model_kwargs = {k: v for k, v in kwargs.items() if k not in _CAPTURE_ONLY}

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    if verbose:
        logger.info("Step 1: Capturing aten-level graphs...")

    # Get model's real output for backward input construction
    with torch.no_grad():
        model_output = model(*args, **model_kwargs)

    output, capture = capture_aten_graphs(
        model, *args, run_backward=run_backward, loss_fn=loss_fn,
        **capture_only, **model_kwargs,
    )

    # --- Forward (all fragments) ---
    forward_outputs_raw = []
    for fi, fg in enumerate(capture.forward_graphs):
        gm = fg.graph_module
        inputs = _build_real_forward_inputs(model, args, gm, example_inputs=fg.example_inputs)
        n_nodes = len(list(gm.graph.nodes))
        suffix = f" (fragment {fi})" if len(capture.forward_graphs) > 1 else ""
        kind = f"forward_{fi}" if len(capture.forward_graphs) > 1 else "forward"
        if verbose:
            logger.info("  Forward graph%s: %d nodes, %d inputs", suffix, n_nodes, len(inputs))

        if verbose:
            logger.info("Step 2: Recording forward%s reference tensors (run 1)...", suffix)
        reference, forward_output_raw = _run_forward(gm, *inputs)
        forward_outputs_raw.append(forward_output_raw)
        if verbose:
            logger.info("  Recorded %d tensors", len(reference))

        if verbose:
            logger.info("Step 3: Re-running forward%s (run 2)...", suffix)
        actual = trace_all_intermediates(gm, *inputs)
        if verbose:
            logger.info("  Recorded %d tensors", len(actual))

        if verbose:
            logger.info("Step 4: Comparing forward%s tensors...", suffix)
        comps = compare_tensors(reference, actual, atol=atol, rtol=rtol)
        fw_result = DumpResult(
            reference_tensors=reference,
            actual_tensors=actual,
            comparisons=comps,
            kind=kind,
        )
        results.append(fw_result)
        if verbose:
            logger.info(fw_result.summary())

        if save_tensors:
            torch.save(reference, str(out_dir / f"{kind}_reference.pt"))
            torch.save(actual, str(out_dir / f"{kind}_actual.pt"))

    # --- Backward (all fragments) ---
    if run_backward:
        # Compute backward-to-forward pairing (backward order is often reversed)
        bw_to_fw: dict[int, int] = {}
        if len(capture.backward_graphs) > 1:
            for bi, bg in enumerate(capture.backward_graphs):
                bw_phs = [n for n in bg.graph_module.graph.nodes if n.op == "placeholder"]
                num_tangents = sum(1 for p in bw_phs if "tangent" in p.name)
                num_saved = len(bw_phs) - num_tangents
                best_fi = None
                for fi in range(len(capture.forward_graphs)):
                    if fi in bw_to_fw.values():
                        continue
                    fg_nodes = capture.forward_graphs[fi].graph_module.graph.nodes
                    fw_out_node = [n for n in fg_nodes if n.op == "output"]
                    if fw_out_node:
                        fw_out_args = fw_out_node[0].args[0]
                        num_fw_returns = len(fw_out_args) if isinstance(fw_out_args, (tuple, list)) else 1
                        if num_fw_returns - num_saved > 0:
                            best_fi = fi
                            break
                if best_fi is not None:
                    bw_to_fw[bi] = best_fi
        elif len(capture.backward_graphs) == 1:
            bw_to_fw[0] = 0

        for bi, bg in enumerate(capture.backward_graphs):
            gm_bw = bg.graph_module
            # Use matching forward output if available (respecting reversed ordering)
            paired_fi = bw_to_fw.get(bi)
            fw_out = forward_outputs_raw[paired_fi] if paired_fi is not None and paired_fi < len(forward_outputs_raw) else None

            if fw_out is not None:
                bw_inputs = _build_backward_inputs_from_forward(
                    fw_out, model_output, gm_bw, bg.example_inputs,
                )
            else:
                bw_inputs = _materialize_inputs(bg.example_inputs)

            suffix = f" (fragment {bi})" if len(capture.backward_graphs) > 1 else ""
            kind = f"backward_{bi}" if len(capture.backward_graphs) > 1 else "backward"
            n_nodes = len(list(gm_bw.graph.nodes))
            if verbose:
                logger.info("  Backward graph%s: %d nodes, %d inputs", suffix, n_nodes, len(bw_inputs))

            if verbose:
                logger.info("Step 5: Recording backward%s reference tensors (run 1)...", suffix)
            bw_reference = trace_all_intermediates(gm_bw, *bw_inputs)
            if verbose:
                logger.info("  Recorded %d tensors", len(bw_reference))

            if verbose:
                logger.info("Step 6: Re-running backward%s (run 2)...", suffix)
            bw_actual = trace_all_intermediates(gm_bw, *bw_inputs)
            if verbose:
                logger.info("  Recorded %d tensors", len(bw_actual))

            if verbose:
                logger.info("Step 7: Comparing backward%s tensors...", suffix)
            bw_comps = compare_tensors(bw_reference, bw_actual, atol=atol, rtol=rtol)
            bw_result = DumpResult(
                reference_tensors=bw_reference,
                actual_tensors=bw_actual,
                comparisons=bw_comps,
                kind=kind,
            )
            results.append(bw_result)
            if verbose:
                logger.info(bw_result.summary())

            if save_tensors:
                torch.save(bw_reference, str(out_dir / f"{kind}_reference.pt"))
                torch.save(bw_actual, str(out_dir / f"{kind}_actual.pt"))

    if not verbose:
        logger.info(" | ".join(r.oneline() for r in results))

    return results


def verify_against_model(
    model: nn.Module,
    *args,
    atol: float = 1e-5,
    rtol: float = 1e-4,
    run_backward: bool = True,
    verbose: bool = True,
    **kwargs,
) -> dict[str, Any]:
    """Verify aten decomposition reproduces the model's output via aot_eager round-trip."""
    results: dict[str, Any] = {"forward": None, "backward": None}

    def _compare_pair(name, ref, act):
        ref_f, act_f = ref.float().cpu(), act.float().cpu()
        if ref_f.numel() == 0:
            return TensorComparison(name=name, matches=True,
                                    ref_shape=list(ref.shape), actual_shape=list(act.shape),
                                    ref_dtype=str(ref.dtype), actual_dtype=str(act.dtype))
        abs_diff = (ref_f - act_f).abs()
        max_abs = abs_diff.max().item()
        mean_abs = abs_diff.mean().item()
        max_rel = (abs_diff / ref_f.abs().clamp(min=1e-12)).max().item()
        ok = torch.allclose(ref_f, act_f, atol=atol, rtol=rtol)
        return TensorComparison(name=name, matches=ok,
                                max_abs_diff=max_abs, mean_abs_diff=mean_abs,
                                max_rel_diff=max_rel,
                                ref_shape=list(ref.shape), actual_shape=list(act.shape),
                                ref_dtype=str(ref.dtype), actual_dtype=str(act.dtype))

    model_ref = copy.deepcopy(model)
    model_comp = copy.deepcopy(model)

    def _clone_args(args_in, need_grad):
        return tuple(
            a.detach().clone().requires_grad_(a.is_floating_point() and need_grad)
            if isinstance(a, torch.Tensor) else a for a in args_in
        )

    ref_args = _clone_args(args, run_backward)
    comp_args = _clone_args(args, run_backward)

    real_output = model_ref(*ref_args, **kwargs)

    torch.compiler.reset()
    compiled_fn = torch.compile(model_comp, backend="aot_eager")
    compiled_output = compiled_fn(*comp_args, **kwargs)

    def _flat(out):
        if isinstance(out, torch.Tensor):
            return [out.detach()]
        if isinstance(out, (tuple, list)):
            return [x.detach() for x in out if isinstance(x, torch.Tensor)]
        return []

    ref_t, comp_t = _flat(real_output), _flat(compiled_output)
    fw_comparisons = [
        _compare_pair(f"output_{i}", ref_t[i], comp_t[i])
        if tuple(ref_t[i].shape) == tuple(comp_t[i].shape)
        else TensorComparison(name=f"output_{i}", matches=False, shape_mismatch=True,
                              ref_shape=list(ref_t[i].shape), actual_shape=list(comp_t[i].shape))
        for i in range(min(len(ref_t), len(comp_t)))
    ]
    results["forward"] = fw_comparisons

    if verbose:
        n_ok = sum(c.matches for c in fw_comparisons)
        logger.info("Forward verification: %d/%d outputs match model", n_ok, len(fw_comparisons))
        for c in fw_comparisons:
            if not c.matches:
                logger.warning("  FAIL: %s", c)

    if run_backward:
        def _backward(out):
            if isinstance(out, torch.Tensor):
                out.sum().backward()
            elif isinstance(out, (tuple, list)):
                s = sum(x.sum() for x in out if isinstance(x, torch.Tensor) and x.is_floating_point())
                s.backward()

        _backward(real_output)
        _backward(compiled_output)

        bw_comparisons = []
        for (rn, rp), (_, cp) in zip(model_ref.named_parameters(), model_comp.named_parameters()):
            if rp.grad is not None and cp.grad is not None:
                bw_comparisons.append(_compare_pair(f"grad_{rn}", rp.grad.detach(), cp.grad.detach()))
            elif rp.grad is None and cp.grad is None:
                pass
            else:
                bw_comparisons.append(TensorComparison(
                    name=f"grad_{rn}", matches=False, missing=True,
                    ref_shape=list(rp.shape), ref_dtype=str(rp.dtype)))

        results["backward"] = bw_comparisons

        if verbose:
            n_ok = sum(c.matches for c in bw_comparisons)
            logger.info("Backward verification: %d/%d gradients match autograd", n_ok, len(bw_comparisons))
            for c in bw_comparisons:
                if not c.matches:
                    logger.warning("  FAIL: %s", c)

    return results


def dump_model_tensors(
    model: nn.Module,
    *args,
    output_path: str = "tensor_dump.pt",
    run_backward: bool = False,
    loss_fn: Callable | None = None,
    **kwargs,
) -> dict[str, dict[str, torch.Tensor]]:
    """Convenience: run model, capture all aten-level intermediates, save to .pt."""
    from torch_graph.export import capture_aten_graphs

    output, capture = capture_aten_graphs(
        model, *args, run_backward=run_backward, loss_fn=loss_fn, **kwargs,
    )

    all_tensors = {}

    for fi, fg in enumerate(capture.forward_graphs):
        key = f"forward_{fi}" if len(capture.forward_graphs) > 1 else "forward"
        inputs = _build_real_forward_inputs(model, args, fg.graph_module)
        all_tensors[key] = trace_all_intermediates(fg.graph_module, *inputs)

    if run_backward:
        for bi, bg in enumerate(capture.backward_graphs):
            key = f"backward_{bi}" if len(capture.backward_graphs) > 1 else "backward"
            bw_inputs = _materialize_inputs(bg.example_inputs)
            all_tensors[key] = trace_all_intermediates(bg.graph_module, *bw_inputs)

    torch.save(all_tensors, output_path)
    for kind, tensors in all_tensors.items():
        logger.info("  %s: %d tensors", kind, len(tensors))
    logger.info("Saved to %s", output_path)
    return all_tensors
