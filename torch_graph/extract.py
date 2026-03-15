"""General-purpose extraction of aten graphs from real training steps.

Works with any PyTorch model/training setup via a simple "recipe" interface.
A recipe is any Python module that defines a ``setup()`` function returning
a dict with ``model``, ``sample_args``, and optionally ``loss_fn``,
``get_batch``, ``optimizer``.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn

logger = logging.getLogger("torch_graph")

from torch_graph.export import capture_aten_graphs, capture_optimizer_aten, export_aten_program, save_step_data, _build_primal_map
from torch_graph.tensor_dump import compute_tensor_stats
from torch_graph.visualizer import GraphVisualizer


def _generate_multi_frag_html(capture, output_dir, prefix, files):
    """Generate one HTML visualization per forward/backward fragment pair."""
    for fi, fg in enumerate(capture.forward_graphs):
        bw = capture.backward_graphs[fi] if fi < len(capture.backward_graphs) else None
        frag_names = capture.per_frag_primal_names[fi] if fi < len(capture.per_frag_primal_names) else []
        frag_map = _build_primal_map(fg.graph_module, capture, frag_primal_names=frag_names)
        html_path = os.path.join(output_dir, f"{prefix}_frag{fi}_graph.html")
        GraphVisualizer(fg).save_html(
            html_path, f"{prefix} aten graph (fragment {fi})",
            source_map=capture.source_map,
            backward_source=bw,
            primal_map=frag_map,
        )
        files.append(html_path)
        logger.info("  Wrote: %s", html_path)


def _do_training_step(model, args, kw, loss_fn, optimizer, step_fn=None):
    """Run one forward+backward+optimizer training step."""
    output = model(*args, **kw)
    if loss_fn is not None:
        loss = loss_fn(output)
    elif isinstance(output, torch.Tensor):
        loss = output.sum()
    else:
        raise ValueError("Model output is not a scalar tensor and no loss_fn provided. "
                         "Pass loss_fn=callable(output)->scalar.")
    loss.backward()
    if step_fn is not None:
        step_fn()
    else:
        optimizer.step()
    if hasattr(optimizer, 'zero_grad'):
        optimizer.zero_grad(set_to_none=True)
    return loss.item()


def _capture_at_step(
    model, cap_args, cap_kw, loss_fn, optimizer, step_num, *,
    triton=False, record_real_tensors=True, record_filter=None,
    capture_optimizer=False, step_fn=None,
):
    """Capture aten graphs + optionally optimizer at the current model state.

    Returns (capture, loss_value).  The optimizer is stepped as part of the
    capture so the model advances by one training step.  When capture_optimizer
    is False, optimizer still runs but without aten tracing.
    """
    _t0 = time.time()
    logger.info("Capturing aten graphs at step %d …", step_num)
    output, capture = capture_aten_graphs(
        model, *cap_args,
        run_backward=True,
        loss_fn=loss_fn,
        record_real_tensors=record_real_tensors,
        record_filter=record_filter,
        triton=triton,
        **cap_kw,
    )
    _t_capture = time.time() - _t0

    if loss_fn is not None:
        loss_value = loss_fn(output).item()
    elif isinstance(output, torch.Tensor):
        loss_value = output.sum().item()
    else:
        loss_value = None

    n_fw_frags = len(capture.forward_graphs)
    n_bw_frags = len(capture.backward_graphs)
    if n_fw_frags > 1:
        logger.info("  Graph breaks: %d forward + %d backward fragments", n_fw_frags, n_bw_frags)
        for i, fg in enumerate(capture.forward_graphs):
            n = len(list(fg.graph_module.graph.nodes))
            logger.info("    FW fragment %d: %d nodes", i, n)
    else:
        fw = capture.forward_graphs[0]
        bw = capture.backward_graphs[0] if capture.backward_graphs else None
        fw_nodes = len(list(fw.graph_module.graph.nodes))
        bw_nodes = len(list(bw.graph_module.graph.nodes)) if bw else 0
        n_fw_int = len(capture.forward_intermediates or {})
        n_bw_int = len(capture.backward_intermediates or {})
        logger.info("  Forward:  %d nodes, %d intermediates recorded", fw_nodes, n_fw_int)
        logger.info("  Backward: %d nodes, %d intermediates recorded", bw_nodes, n_bw_int)

    logger.debug("  Capture time: %.1fs", _t_capture)
    if loss_value is not None:
        logger.info("  Loss at capture step: %.4f", loss_value)

    if capture_optimizer:
        # capture optimizer aten ops (this also steps the optimizer)
        param_order = [(n, p) for n, p in model.named_parameters()]
        pre_step_params = {n: p.data.clone().detach() for n, p in param_order}
        gradients = {}
        for n, p in param_order:
            if p.grad is not None:
                gradients[n] = p.grad.clone().detach()

        # Snapshot optimizer state (standard torch.optim.Optimizer interface)
        opt_state_before = {}
        if hasattr(optimizer, 'state'):
            for n, p in param_order:
                if p in optimizer.state:
                    s = optimizer.state[p]
                    opt_state_before[n] = {
                        k: v.clone().detach() if isinstance(v, torch.Tensor) else v
                        for k, v in s.items()
                    }

        # Extract param group info (if standard interface available)
        param_set = {id(p): n for n, p in param_order}
        param_groups_info = []
        if hasattr(optimizer, 'param_groups'):
            for group in optimizer.param_groups:
                info = {k: v for k, v in group.items() if k != "params"}
                info["param_names"] = [param_set[id(p)] for p in group["params"]]
                param_groups_info.append(info)

        logger.info("  Extracting optimizer aten ops via torch.compile …")
        _t0 = time.time()
        opt_capture = capture_optimizer_aten(
            optimizer,
            record_real_tensors=record_real_tensors,
            param_name_map={id(p): n for n, p in param_order},
            step_fn=step_fn,
        )
        _t_opt = time.time() - _t0
        logger.debug("  Optimizer capture time: %.1fs", _t_opt)
        if hasattr(optimizer, 'zero_grad'):
            optimizer.zero_grad(set_to_none=True)

        post_step_params = {n: p.data.clone().detach() for n, p in param_order}
        n_changed = sum(
            1 for n in pre_step_params
            if not torch.equal(pre_step_params[n], post_step_params[n])
        )

        og = opt_capture.forward_graphs[0] if opt_capture.forward_graphs else None
        opt_nodes = len(list(og.graph_module.graph.nodes)) if og else 0

        capture.optimizer_data = {
            "class": type(optimizer).__name__,
            "param_groups": param_groups_info,
            "state_before": opt_state_before,
            "pre_step_params": pre_step_params,
            "gradients": gradients,
            "post_step_params": post_step_params,
        }
        capture.optimizer_capture = opt_capture
        logger.info("  Optimizer: %s, %d aten ops, %d/%d params changed",
                     type(optimizer).__name__, opt_nodes, n_changed, len(pre_step_params))
    else:
        # Still advance the optimizer to keep model state consistent
        if step_fn is not None:
            step_fn()
        else:
            optimizer.step()
        if hasattr(optimizer, 'zero_grad'):
            optimizer.zero_grad(set_to_none=True)

    return capture, loss_value


def _to_device(tensors, device):
    """Move a tuple/list of tensors to *device*."""
    return tuple(
        t.to(device) if isinstance(t, torch.Tensor) else t for t in tensors
    )


def extract_training_step(
    model: nn.Module,
    sample_args: tuple,
    sample_kwargs: dict | None = None,
    loss_fn: Callable | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    warmup_steps: int = 0,
    steps: list[int] | None = None,
    get_batch: Callable[[int], tuple[tuple, dict]] | None = None,
    output_dir: str = "outputs",
    prefix: str = "model",
    max_intermediates_mb: float | None = None,
    storage_dtype: torch.dtype | None = None,
    device: str = "cpu",
    triton: bool = False,
    record_real_tensors: bool = True,
    record_filter: dict | None = None,
    capture_optimizer: bool = False,
    graph_only: bool = False,
    step_fn: Callable | None = None,
) -> dict[str, Any]:
    """Extract aten-level forward+backward graphs from real training steps.

    Single-step mode (default): runs warmup_steps real iterations then captures.
    Multi-step mode (steps=[0,5,10]): captures at each listed step, producing
    one .py graph file + one .pt data file per step.

    For non-standard optimizers (without ``param_groups``/``zero_grad``),
    pass *step_fn* — a zero-arg callable that performs one optimizer step.
    This is forwarded to both warmup training steps and
    ``capture_optimizer_aten``.

    Returns dict with capture(s), loss_values, and list of output file paths.
    """
    sample_kwargs = sample_kwargs or {}

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    target_device = torch.device(device)

    if triton and not torch.cuda.is_available():
        raise RuntimeError("--triton requires CUDA but no GPU is available")

    model.to(target_device)
    sample_args = _to_device(sample_args, target_device)
    sample_kwargs = {k: v.to(target_device) if isinstance(v, torch.Tensor) else v
                     for k, v in sample_kwargs.items()}

    model.train()

    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    if steps is None:
        steps = [warmup_steps]
    sorted_steps = sorted(set(steps))
    multi = len(sorted_steps) > 1

    _orig_get_batch = get_batch
    if _orig_get_batch is not None and str(target_device) != "cpu":
        def get_batch(step):
            args, kw = _orig_get_batch(step)
            args = _to_device(args, target_device)
            kw = {k: v.to(target_device) if isinstance(v, torch.Tensor) else v
                  for k, v in kw.items()}
            return args, kw

    # -----------------------------------------------------------------------------
    step_captures: dict[int, Any] = {}
    loss_values: dict[int, float | None] = {}
    current_step = 0

    for target_step in sorted_steps:
        n_train = target_step - current_step
        if n_train > 0:
            logger.info("Training %d step(s) to reach step %d …", n_train, target_step)
        for step in range(current_step, target_step):
            args, kw = get_batch(step) if get_batch else (sample_args, sample_kwargs)
            loss_val = _do_training_step(model, args, kw, loss_fn, optimizer, step_fn=step_fn)
            progress_every = max(1, n_train // 5)
            done = step - current_step + 1
            if done % progress_every == 0 or step == target_step - 1:
                logger.info("  step %d  loss=%.4f", step, loss_val)

        cap_args, cap_kw = get_batch(target_step) if get_batch else (sample_args, sample_kwargs)
        capture, loss_value = _capture_at_step(
            model, cap_args, cap_kw, loss_fn, optimizer, target_step,
            triton=triton,
            record_real_tensors=record_real_tensors,
            record_filter=record_filter,
            capture_optimizer=capture_optimizer,
            step_fn=step_fn,
        )
        step_captures[target_step] = capture
        loss_values[target_step] = loss_value
        current_step = target_step + 1

    # -----------------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    files = []

    first_capture = step_captures[sorted_steps[0]]
    py_path = os.path.join(output_dir, f"{prefix}_aten.py")

    # Triton kernel mapping (if capture succeeded)
    kernel_mapping = None
    if first_capture.triton_capture:
        from torch_graph.triton import build_kernel_node_map, save_triton_kernels
        kernel_mapping = build_kernel_node_map(first_capture, first_capture.triton_capture)
        logger.info("  %s", kernel_mapping.summary())
        saved_kernels = save_triton_kernels(
            first_capture.triton_capture, output_dir, prefix=f"{prefix}_"
        )
        files.extend(str(p) for p in saved_kernels)
        logger.info("  Saved %d Triton kernel files", len(saved_kernels))

    _save_kw = {}
    if max_intermediates_mb is not None:
        _save_kw["max_intermediates_mb"] = max_intermediates_mb
    if storage_dtype is not None:
        _save_kw["storage_dtype"] = storage_dtype
    _export_kw = {}
    if kernel_mapping is not None:
        _export_kw["kernel_map"] = kernel_mapping.node_to_kernel

    _is_multi_frag = len(first_capture.forward_graphs) > 1

    _t0 = time.time()
    if graph_only:
        # Skip .pt weights — only generate .py script + .html graph
        export_aten_program(first_capture, py_path, weights_path="/dev/null",
                            **_save_kw, **_export_kw)
        files.append(py_path)
        logger.info("  Wrote: %s", py_path)
    elif multi:
        export_aten_program(first_capture, py_path, available_steps=sorted_steps,
                            **_save_kw, **_export_kw)
        files.append(py_path)
        logger.info("  Wrote: %s", py_path)

        # Save GM pickles once with stable aten naming (graph is step-independent)
        gm_base = os.path.join(output_dir, f"{prefix}_aten")
        if _is_multi_frag:
            for _gi, _fg in enumerate(first_capture.forward_graphs):
                torch.save(_fg.graph_module, f"{gm_base}_gm_fw_{_gi}.pt")
            for _gi, _bg in enumerate(first_capture.backward_graphs):
                torch.save(_bg.graph_module, f"{gm_base}_gm_bw_{_gi}.pt")
        else:
            if first_capture.forward_graphs:
                torch.save(first_capture.forward_graphs[0].graph_module, f"{gm_base}_gm_fw.pt")
            if first_capture.backward_graphs:
                torch.save(first_capture.backward_graphs[0].graph_module, f"{gm_base}_gm_bw.pt")

        for step, cap in step_captures.items():
            pt_path = os.path.join(output_dir, f"{prefix}_step{step}.pt")
            if step != sorted_steps[0]:
                if cap.optimizer_capture:
                    cap.optimizer_capture.forward_real_inputs = None
                    cap.optimizer_capture.forward_real_output = None
            save_step_data(cap, pt_path, save_gm=False, **_save_kw)
            files.append(pt_path)
            logger.info("  Wrote: %s", pt_path)
    else:
        export_aten_program(first_capture, py_path, **_save_kw, **_export_kw)
        files.append(py_path)
        pt_path = py_path.replace(".py", ".pt")
        if os.path.exists(pt_path):
            files.append(pt_path)
        logger.info("  Wrote: %s", py_path)

    _t_export = time.time() - _t0
    logger.debug("  Export time: %.1fs", _t_export)

    if _is_multi_frag:
        _t0 = time.time()
        _generate_multi_frag_html(first_capture, output_dir, prefix, files)
        _t_html = time.time() - _t0
        logger.debug("  HTML time: %.1fs (%d fragments)", _t_html, len(first_capture.forward_graphs))
    else:
        fw = first_capture.forward_graphs[0]
        bw = first_capture.backward_graphs[0] if first_capture.backward_graphs else None
        html_path = os.path.join(output_dir, f"{prefix}_graph.html")

        _t0 = time.time()
        stats = compute_tensor_stats(first_capture.forward_intermediates) if first_capture.forward_intermediates else None
        bw_stats = compute_tensor_stats(first_capture.backward_intermediates) if first_capture.backward_intermediates else None
        _t_stats = time.time() - _t0
        n_fw_stats = len(first_capture.forward_intermediates or {})
        n_bw_stats = len(first_capture.backward_intermediates or {})
        logger.debug("  Stats time: %.1fs (%d FW + %d BW tensors)", _t_stats, n_fw_stats, n_bw_stats)

        vis_kw = {}
        if kernel_mapping is not None:
            from torch_graph.op_dump import _build_triton_call_map
            call_map = _build_triton_call_map(first_capture)
            vis_kw["kernel_map"] = call_map if call_map else kernel_mapping.node_to_kernel
            vis_kw["kernel_details"] = kernel_mapping.kernel_details

        # Build primal map for human-readable parameter names in the viewer
        primal_map = _build_primal_map(fw.graph_module, first_capture)

        _t0 = time.time()
        GraphVisualizer(fw).save_html(html_path, f"{prefix} aten graph", tensor_stats=stats,
                                       source_map=first_capture.source_map,
                                       backward_source=bw, bw_tensor_stats=bw_stats,
                                       primal_map=primal_map,
                                       **vis_kw)
        _t_html = time.time() - _t0
        files.append(html_path)
        logger.info("  Wrote: %s", html_path)
        logger.debug("  HTML time: %.1fs", _t_html)

    if multi:
        return {
            "captures": step_captures,
            "loss_values": loss_values,
            "files": files,
        }
    return {
        "capture": first_capture,
        "loss_value": loss_values[sorted_steps[0]],
        "files": files,
    }


def extract_function(
    fn: Callable,
    *args,
    run_backward: bool = False,
    loss_fn: Callable | None = None,
    output_dir: str = "outputs",
    prefix: str = "fn",
    max_intermediates_mb: float | None = None,
    storage_dtype: torch.dtype | None = None,
    device: str = "cpu",
    record_real_tensors: bool = True,
    record_filter: dict | None = None,
    param_names: dict[int, str] | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Extract aten-level graphs from an arbitrary function call.

    Unlike ``extract_training_step`` (which requires an nn.Module and
    orchestrates a full training step), this wraps **any** callable that
    uses PyTorch operations — a free function, a lambda, a single layer's
    forward method, or even a closure over model weights.

    This is useful when:
      - The full model is too large to capture at once.
      - You only care about a specific sub-computation (e.g. one attention
        head, one loss function, a custom kernel).
      - The model has graph breaks that prevent full capture, but a smaller
        piece compiles cleanly.
      - The computation isn't wrapped in an ``nn.Module`` at all.

    Examples::

        # Capture a standalone function
        def my_fn(x, w):
            return torch.nn.functional.linear(torch.relu(x), w)
        result = extract_function(my_fn, x, w, prefix="relu_linear")

        # Capture one layer of a model
        result = extract_function(
            model.transformer.layers[0],  # nn.Module is also callable
            hidden_states,
            run_backward=True,
            prefix="layer0",
        )

        # Capture with a loss function
        result = extract_function(
            model, x,
            run_backward=True,
            loss_fn=lambda out: out.sum(),
            prefix="model_with_loss",
        )

    Args:
        fn: Any callable that uses PyTorch ops. Can be a function,
            lambda, nn.Module, or bound method.
        *args: Positional arguments to ``fn``.
        run_backward: If True, also capture the backward graph.
        loss_fn: Loss function for backward.  Required when
            ``run_backward=True`` and ``fn`` doesn't return a scalar.
        output_dir: Directory for exported files.
        prefix: Filename prefix for outputs.
        max_intermediates_mb: Cap intermediate tensor storage.
        storage_dtype: Store tensors in this dtype.
        device: Device to run on.
        record_real_tensors: Record intermediate tensor values.
        record_filter: Selective recording filter.
        param_names: Optional mapping from arg index to a human-readable
            name, e.g. ``{0: "hidden_states", 1: "weight"}``.
            These names appear in the exported script's header.
        **kwargs: Keyword arguments forwarded to ``fn``.

    Returns:
        Dict with ``capture`` (AtenCapture), ``output``, ``files``.
    """
    target_device = torch.device(device)

    # Move tensor args to device
    args = tuple(
        a.to(target_device) if isinstance(a, torch.Tensor) else a for a in args
    )
    kwargs = {
        k: v.to(target_device) if isinstance(v, torch.Tensor) else v
        for k, v in kwargs.items()
    }

    # If fn is an nn.Module, move it too
    if isinstance(fn, nn.Module):
        fn.to(target_device)
    # Move closure tensors for plain functions
    elif hasattr(fn, '__closure__') and fn.__closure__:
        for cell in fn.__closure__:
            try:
                obj = cell.cell_contents
            except ValueError:
                continue
            if isinstance(obj, torch.Tensor):
                cell.cell_contents = obj.to(target_device)

    # Set custom param_names on the capture if provided
    # (also works for nn.Module where we want to override names)
    output, capture = capture_aten_graphs(
        fn, *args,
        run_backward=run_backward,
        loss_fn=loss_fn,
        record_real_tensors=record_real_tensors,
        record_filter=record_filter,
        **kwargs,
    )

    if not capture.forward_graphs:
        raise RuntimeError(
            "No aten graphs captured. The function may use ops that "
            "torch.compile cannot trace (e.g. LSTM, data-dependent control flow)."
        )

    # Apply user-provided param_names to the primal mapping
    if param_names and not capture.primal_names:
        # Build primal_names list where user-named args are labeled
        fw = capture.forward_graphs[0]
        placeholders = [n for n in fw.graph_module.graph.nodes if n.op == "placeholder"]
        names = []
        for i, ph in enumerate(placeholders):
            if i in param_names:
                names.append(param_names[i])
            else:
                names.append(None)
        # Store as primal_names (None entries → unnamed inputs)
        capture.primal_names = names

    # Export
    os.makedirs(output_dir, exist_ok=True)
    files = []

    _save_kw = {}
    if max_intermediates_mb is not None:
        _save_kw["max_intermediates_mb"] = max_intermediates_mb
    if storage_dtype is not None:
        _save_kw["storage_dtype"] = storage_dtype

    py_path = os.path.join(output_dir, f"{prefix}_aten.py")
    export_aten_program(capture, py_path, **_save_kw)
    files.append(py_path)
    pt_path = py_path.replace(".py", ".pt")
    if os.path.exists(pt_path):
        files.append(pt_path)

    if len(capture.forward_graphs) > 1:
        _generate_multi_frag_html(capture, output_dir, prefix, files)
    else:
        fw = capture.forward_graphs[0]
        bw = capture.backward_graphs[0] if capture.backward_graphs else None
        html_path = os.path.join(output_dir, f"{prefix}_graph.html")
        stats = compute_tensor_stats(capture.forward_intermediates) if capture.forward_intermediates else None
        bw_stats = compute_tensor_stats(capture.backward_intermediates) if capture.backward_intermediates else None

        fn_primal_map = _build_primal_map(fw.graph_module, capture)
        GraphVisualizer(fw).save_html(
            html_path, f"{prefix} aten graph",
            tensor_stats=stats,
            source_map=capture.source_map,
            backward_source=bw,
            bw_tensor_stats=bw_stats,
            primal_map=fn_primal_map,
        )
        files.append(html_path)

    return {
        "capture": capture,
        "output": output,
        "files": files,
    }


def load_recipe(recipe_path: str, setup_fn: str = "setup") -> dict[str, Any]:
    """Import a recipe module and call its setup function.

    A recipe is a Python file that defines::

        def setup() -> dict:
            return {
                "model": nn.Module,           # required
                "sample_args": tuple,          # required – one batch
                "loss_fn": callable | None,    # optional
                "get_batch": callable | None,  # optional  (step) -> (args, kw)
                "optimizer": Optimizer | None,  # optional
                "step_fn": callable | None,    # optional — custom optimizer step
            }

    A recipe may also define alternative setup functions (e.g.
    ``setup_sft()``, ``setup_rl()``) which can be selected via
    *setup_fn*.

    Returns:
        The dict returned by the setup function.
    """
    import importlib.util

    path = Path(recipe_path).resolve()
    spec = importlib.util.spec_from_file_location("_recipe", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, setup_fn):
        raise AttributeError(
            f"Recipe {recipe_path} must define a {setup_fn}() function"
        )
    return getattr(mod, setup_fn)()
