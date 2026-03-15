"""Drop-in replacement for torch.compile using captured aten graphs.

Provides install(model, aten_module) which swaps the model's forward/backward
with hand-editable aten-level graphs captured by torch_graph.export.

Usage:
    import my_model_aten as aten_mod
    from torch_graph.install import install
    install(model, aten_mod)
    # model(x) now runs aten_mod.forward/backward instead of the original

Also provides install_optimizer(optimizer, aten_module) for replacing
@torch.compile'd optimizer step functions.
"""

from __future__ import annotations

import functools
import inspect
import logging
import operator
import re
import types
from typing import Any, Callable

import torch
import torch.nn as nn

logger = logging.getLogger("torch_graph")


# -----------------------------------------------------------------------------
# Model install
# -----------------------------------------------------------------------------


def _parse_param_paths(aten_module) -> list[str]:
    """Extract parameter paths from the aten module's docstring/source.

    The forward function has parameter names like wte_weight, blocks_0_ln_1_weight.
    The module docstring has the mapping back to nn.Module paths:
        wte_weight → self.wte.weight
    We parse both formats from the "Parameter mapping:" section.
    """
    try:
        src = inspect.getsource(aten_module)
    except (TypeError, OSError):
        src = ''

    # Fallback: read the file directly when inspect.getsource fails
    if not src and hasattr(aten_module, '__file__'):
        try:
            with open(aten_module.__file__) as f:
                src = f.read()
        except Exception as e:
            logger.debug("Failed to read aten module source from %s: %s", aten_module.__file__, e)

    param_paths = []

    in_param_section = False
    for line in src.split('\n'):
        stripped = line.strip()
        if 'Parameter mapping:' in stripped:
            in_param_section = True
            continue
        if in_param_section:
            if stripped.startswith('"""') or stripped.startswith("'''"):
                break
            if not stripped or stripped.startswith('#'):
                continue
            # "varname → self.path  [shape]" format
            m = re.match(r'\s*\S+\s+→\s+self\.(\S+)\s+', line)
            if m:
                param_paths.append(m.group(1))
                continue
            # "self.path  [shape]" format
            m = re.match(r'\s*self\.(\S+)\s+\[', line)
            if m:
                param_paths.append(m.group(1))
                continue

    return param_paths


@functools.lru_cache(maxsize=512)
def _path_getter(path: str) -> Callable[[Any], Any]:
    """Compile a dotted attribute path once for repeated live lookups."""
    return operator.attrgetter(path)


def _split_parent_attr(path: str) -> tuple[str, str]:
    """Split 'a.b.c' into ('a.b', 'c'); top-level attrs return ('', attr)."""
    parent_path, _, attr = path.rpartition('.')
    if not attr:
        return '', path
    return parent_path, attr


def _resolve_param(model: nn.Module, path: str) -> torch.Tensor:
    """Resolve a dotted path like 'blocks.0.ln_1.weight' to the actual parameter."""
    return _path_getter(path)(model)


def _make_live_attr_getter(model: nn.Module, path: str) -> Callable[[], Any]:
    """Return a cheap live accessor for a model attr used by an installed graph."""
    parent_path, attr = _split_parent_attr(path)
    if not parent_path:
        return lambda model=model, attr=attr: getattr(model, attr)

    parent_getter = _path_getter(parent_path)
    return lambda model=model, parent_getter=parent_getter, attr=attr: getattr(
        parent_getter(model), attr
    )


def _set_live_attr(owner: Any, attr: str, value: Any) -> None:
    """Write a value to a model attribute, preserving tensor identity when possible.

    Uses in-place copy_ first so that external references to the buffer tensor
    (e.g. from optimizer param groups) remain valid.  Falls back to setattr
    when shapes are incompatible.
    """
    current = getattr(owner, attr)
    if isinstance(current, torch.Tensor) and isinstance(value, torch.Tensor):
        try:
            with torch.no_grad():
                current.copy_(value)
            return
        except Exception:
            pass  # shape mismatch — fall through to setattr
    setattr(owner, attr, value)


def _make_buffer_writer(model: nn.Module, path: str) -> Callable[[Any], None]:
    """Return a writer for a potentially-mutated buffer captured by aot_autograd."""
    parent_path, attr = _split_parent_attr(path)
    if not parent_path:
        return lambda value, model=model, attr=attr: _set_live_attr(model, attr, value)

    parent_getter = _path_getter(parent_path)
    return lambda value, model=model, parent_getter=parent_getter, attr=attr: _set_live_attr(
        parent_getter(model), attr, value
    )


def _set_buffer(model: nn.Module, path: str, value: torch.Tensor) -> None:
    """Write a value back to a buffer at a dotted path like 'bn.running_mean'."""
    parent_path, attr = _split_parent_attr(path)
    owner = model if not parent_path else _path_getter(parent_path)(model)
    _set_live_attr(owner, attr, value)


def _extract_forward_arg_names(forward: Callable) -> list[str]:
    """Return named forward parameters in call order for kwargs normalization."""
    return [
        p.name for p in inspect.signature(forward).parameters.values()
        if p.name != 'self' and p.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    ]


def _normalize_user_inputs(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    orig_param_names: list[str],
) -> tuple[Any, ...]:
    """Reorder kwargs into the original forward signature order."""
    if kwargs and orig_param_names:
        kwargs = dict(kwargs)
        ordered = list(args)
        for name in orig_param_names[len(args):]:
            if name in kwargs:
                ordered.append(kwargs.pop(name))
        if kwargs:
            ordered.extend(kwargs.values())
        return tuple(ordered)
    if kwargs:
        return args + tuple(kwargs.values())
    return args


class _SymIntSpec:
    """Slot spec for a SymInt: derives its value at runtime from a tensor's shape.

    When dynamic=True, aot_autograd inserts SymInt placeholders (e.g. ``input_0``)
    for dimensions that vary.  At runtime we need to extract the actual dimension
    from the corresponding tensor rather than using a frozen concrete value.
    """
    __slots__ = ('tensor_user_idx', 'dim_idx')

    def __init__(self, tensor_user_idx: int, dim_idx: int):
        self.tensor_user_idx = tensor_user_idx
        self.dim_idx = dim_idx


def _detect_symint_slots(
    fw_param_names: list[str],
    aten_forward: Callable,
    param_paths: list[str],
) -> dict[str, tuple[str, int]]:
    """Detect SymInt parameters and map each to (tensor_param_name, dim_index).

    SymInt params have no type annotation and aren't model params.
    Each SymInt's symbolic name (e.g. "s77") appears in a tensor param's
    annotation like "float32[s77, 8]".

    Matching is greedy by position proximity: each SymInt is mapped to the
    first symbolic dim in the nearest subsequent tensor parameter.  This works
    because aot_autograd places each SymInt immediately before the tensor
    whose shape it describes.
    """
    sig = inspect.signature(aten_forward)
    param_underscores = {path.replace('.', '_') for path in param_paths}

    # Collect annotations and find unannotated non-param slots (candidate SymInts)
    annotations: dict[str, str] = {}
    symint_candidates: list[str] = []
    for p in sig.parameters.values():
        if p.annotation != inspect.Parameter.empty:
            annotations[p.name] = str(p.annotation).strip('"\'')
        elif p.name not in param_underscores:
            symint_candidates.append(p.name)

    if not symint_candidates:
        return {}

    # For each SymInt candidate, find the symbolic dim name from the exported
    # parameter mapping docstring. The SymInt param is named like "input_0"
    # and corresponds to a symbolic dim like "s77". We find "s77" by looking
    # at tensor annotations that contain non-numeric dimension components.
    #
    # Strategy: find tensor annotations with symbolic dims, then match each
    # symbolic dim to the SymInt that precedes the tensor in slot order.
    symint_map: dict[str, tuple[str, int]] = {}

    # Match SymInt candidates to symbolic dims by position proximity:
    # Each SymInt immediately precedes its tensor in the forward signature.
    # When multiple SymInts precede the same tensor, they map to successive
    # symbolic dims in order (e.g. two SymInts before a [s33, s50, 32] tensor
    # map to s33 at dim 0 and s50 at dim 1 respectively).
    claimed_dims: set[tuple[str, int]] = set()  # (tensor_name, dim_idx)

    for si_name in symint_candidates:
        si_pos = fw_param_names.index(si_name)
        # Look at the next param(s) after this SymInt for a tensor with symbolic dims
        for tensor_name, ann in annotations.items():
            tensor_pos = fw_param_names.index(tensor_name) if tensor_name in fw_param_names else -1
            if tensor_pos <= si_pos:
                continue
            m = re.match(r'\w+\[(.*)\]', ann)
            if not m:
                continue
            dims = [d.strip() for d in m.group(1).split(',')]
            for dim_idx, dim in enumerate(dims):
                if dim and not dim.isdigit() and (tensor_name, dim_idx) not in claimed_dims:
                    symint_map[si_name] = (tensor_name, dim_idx)
                    claimed_dims.add((tensor_name, dim_idx))
                    break
            if si_name in symint_map:
                break

    return symint_map


def _build_slot_specs(
    model: nn.Module,
    fw_param_names: list[str],
    param_paths: list[str],
    symint_map: dict[str, tuple[str, int]] | None = None,
) -> tuple[list[int | Callable[[], Any] | _SymIntSpec], int]:
    """Compile model-input slots once so installed forwards stay lightweight.

    symint_map: maps SymInt param names to (tensor_param_name, dim_index).
    When present, SymInt slots auto-derive their value from tensor shapes.
    """
    param_path_to_underscore = {path.replace('.', '_'): path for path in param_paths}
    symint_map = symint_map or {}

    # First pass: classify each forward parameter as one of three types:
    #   - Callable:    model param/buffer → live accessor (reads current value)
    #   - int:         user input → index into the args tuple
    #   - None:        SymInt → placeholder, resolved in second pass
    slot_specs: list[int | Callable[[], Any] | _SymIntSpec] = []
    next_user_idx = 0
    for name in fw_param_names:
        path = param_path_to_underscore.get(name)
        if path is not None:
            slot_specs.append(_make_live_attr_getter(model, path))
        elif name in symint_map:
            slot_specs.append(None)  # placeholder, filled in second pass
        else:
            slot_specs.append(next_user_idx)
            next_user_idx += 1

    # Second pass: resolve SymInt placeholders (None entries).
    # Each SymInt derives its value from a user-input tensor's shape dim.
    # We need user indices assigned first (above) to know which tensor to read.
    name_to_user_idx: dict[str, int] = {}
    for i, name in enumerate(fw_param_names):
        spec = slot_specs[i]
        if isinstance(spec, int):
            name_to_user_idx[name] = spec

    for i, name in enumerate(fw_param_names):
        if slot_specs[i] is not None:
            continue  # already resolved (param getter or user input)
        tensor_name, dim_idx = symint_map[name]
        tensor_user_idx = name_to_user_idx.get(tensor_name)
        if tensor_user_idx is not None:
            slot_specs[i] = _SymIntSpec(tensor_user_idx, dim_idx)
        else:
            # SymInt's tensor is a model param, not a user input.
            # Treat the SymInt itself as a user input (rare edge case).
            slot_specs[i] = next_user_idx
            next_user_idx += 1

    return slot_specs, next_user_idx


def _assemble_inputs(
    user_inputs: tuple[Any, ...],
    slot_specs: list[int | Callable[[], Any] | _SymIntSpec],
    min_user_inputs: int,
) -> list[Any]:
    """Assemble aten inputs from user args and precompiled live model accessors."""
    if len(user_inputs) < min_user_inputs:
        raise RuntimeError(
            f"Not enough user inputs: expected at least {min_user_inputs}, got {len(user_inputs)}"
        )

    result = []
    for spec in slot_specs:
        if isinstance(spec, int):
            result.append(user_inputs[spec])
        elif isinstance(spec, _SymIntSpec):
            result.append(user_inputs[spec.tensor_user_idx].shape[spec.dim_idx])
        else:
            result.append(spec())
    return result


def _validate_shapes(model: nn.Module, param_paths: list[str],
                     aten_forward: Callable, label: str = "model") -> None:
    """Check that model param shapes match the aten forward's type annotations.

    Matches by name (fc1.weight -> fc1_weight), not position, since
    aot_autograd may interleave user inputs at arbitrary positions.
    """
    sig = inspect.signature(aten_forward)
    ann_map = {
        p.name: str(p.annotation).strip('"\'')
        for p in sig.parameters.values()
        if p.annotation != inspect.Parameter.empty
    }
    errors = []
    for path in param_paths:
        ann_str = ann_map.get(path.replace('.', '_'))
        if ann_str is None:
            continue
        m = re.match(r'(\w+)\[([\d,\s]+)\]', ann_str)
        if m:
            expected = tuple(int(x.strip()) for x in m.group(2).split(','))
            actual = tuple(_resolve_param(model, path).shape)
            if expected != actual:
                errors.append(f"  {path}: expected shape {list(expected)}, got {list(actual)}")
    if errors:
        raise ValueError(f"Shape mismatch in {label} install:\n" + "\n".join(errors))


def build_aten_forward(
    model: nn.Module,
    aten_module: types.ModuleType,
    *,
    param_paths: list[str] | None = None,
    orig_forward_params: list[str] | None = None,
    num_real_outputs: int = 1,
    num_mutations: int = 0,
    mutated_buffers: list[str] | None = None,
    validate: bool = True,
    user_input_order: list[int] | None = None,
    compile: bool = False,
) -> Callable:
    """Build an aten forward callable without modifying model.forward.

    Returns a function(*args, **kwargs) -> output that reads model parameters
    live and runs aten forward/backward via autograd.Function. Backward works
    automatically when loss.backward() is called.

    When *compile* is True, the aten forward and backward functions are wrapped
    with ``torch.compile(backend="inductor")`` so that Inductor fuses the aten
    ops into Triton kernels.  This gives near-``torch.compile(model)``
    performance while keeping the aten .py file as the editable source of truth.
    First call incurs a compilation overhead; subsequent calls run fused kernels.

    Used by multi-variant dispatch where each call pattern gets its own
    aten graph. Also used by install() for single-variant monkey-patching.
    """
    aten_forward = aten_module.forward
    aten_backward = getattr(aten_module, 'backward', None)

    if compile:
        aten_forward = torch.compile(aten_forward, backend="inductor")
        if aten_backward is not None:
            aten_backward = torch.compile(aten_backward, backend="inductor")
    _mutated_buffers = mutated_buffers or []

    if param_paths is None:
        param_paths = _parse_param_paths(aten_module)

    if not param_paths:
        raise ValueError(
            "Could not determine parameter paths from aten module. "
            "Pass param_paths= explicitly."
        )

    if validate:
        _validate_shapes(model, param_paths, aten_forward, label=type(model).__name__)

    sig = inspect.signature(aten_forward)
    fw_param_names = list(sig.parameters.keys())
    symint_map = _detect_symint_slots(fw_param_names, aten_forward, param_paths)
    slot_specs, min_user_inputs = _build_slot_specs(model, fw_param_names, param_paths, symint_map)
    buffer_writers = [_make_buffer_writer(model, path) for path in _mutated_buffers]

    if orig_forward_params is None:
        orig_forward_params = _extract_forward_arg_names(model.forward)

    # If user_input_order is provided, remap user indices in slot_specs.
    # aot_autograd may reorder user inputs vs the forward signature.
    # user_input_order[i] = position in the original forward call for the i-th
    # user input slot in the aten graph.
    if user_input_order:
        _uid = 0
        for j, spec in enumerate(slot_specs):
            if isinstance(spec, int):
                if _uid < len(user_input_order):
                    slot_specs[j] = user_input_order[_uid]
                _uid += 1
        min_user_inputs = max(user_input_order) + 1 if user_input_order else min_user_inputs

    # Derive expected forward return length from backward signature when available.
    # backward receives (saved..., grad_outputs...), so:
    #   num_saved = len(backward_params) - num_real_outputs
    _expected_fw_len: int | None = None
    if aten_backward is not None:
        _bw_sig = inspect.signature(aten_backward)
        # Only validate when backward has explicit named params (not *args).
        _bw_params = [
            p for p in _bw_sig.parameters.values()
            if p.name != "self"
            and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]
        if _bw_params:
            _num_saved = len(_bw_params) - num_real_outputs
            if _num_saved >= 0:
                _expected_fw_len = num_mutations + num_real_outputs + _num_saved

    class _AtenGraph(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *all_inputs):
            result = aten_forward(*all_inputs)
            if isinstance(result, tuple):
                # Validate forward return length against expected layout.
                if _expected_fw_len is not None and len(result) != _expected_fw_len:
                    _n_saved = _expected_fw_len - num_mutations - num_real_outputs
                    raise RuntimeError(
                        f"Aten forward returned {len(result)} values but expected "
                        f"{_expected_fw_len} (num_mutations={num_mutations} + "
                        f"num_real_outputs={num_real_outputs} + "
                        f"saved_for_backward={_n_saved}). "
                        f"Did you edit the forward return statement?"
                    )
                # ── Forward output layout (set by aot_autograd) ──────────
                # [buffer_mutations..., real_outputs..., saved_for_backward...]
                #
                # Buffer mutations: functionalized in-place ops like
                #   BatchNorm's running_mean/var.  Must be written back.
                # Real outputs: what the original model.forward() returned.
                # Saved for backward: tensors + SymInts needed by backward().
                mut_vals = result[:num_mutations]
                real_out = result[num_mutations:num_mutations + num_real_outputs]
                saved = result[num_mutations + num_real_outputs:]

                # Write mutated buffer values back to the live model
                for write_buffer, val in zip(buffer_writers, mut_vals):
                    write_buffer(val)

                # ctx.save_for_backward only accepts tensors.
                # Non-tensor saved values (SymInts) are stored separately.
                tensor_saved = []
                non_tensor_saved = {}
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
            if aten_backward is None:
                raise RuntimeError("No backward() function in aten module")

            # Rebuild the saved values list, interleaving tensors and non-tensors
            # back into their original positions.
            tensors = list(ctx.saved_tensors)
            saved = []
            t_idx = 0
            for i in range(ctx._num_saved):
                if i in ctx._non_tensor_saved:
                    saved.append(ctx._non_tensor_saved[i])
                else:
                    saved.append(tensors[t_idx])
                    t_idx += 1

            # ── IMPORTANT: aot_autograd backward signature ordering ──────
            # The aten backward function expects: (non_tensors..., tensors..., tangents...)
            # NOT the original interleaved order.  aot_autograd places SymInts
            # (non-tensor saved values) before all tensor saved values in the
            # backward graph's placeholder list.  We must match that layout.
            if ctx._non_tensor_saved:
                non_tensors = [saved[i] for i in sorted(ctx._non_tensor_saved)]
                tensors_only = [v for v in saved if isinstance(v, torch.Tensor)]
                saved = non_tensors + tensors_only

            bw_result = aten_backward(*saved, *grad_outputs)

            if not isinstance(bw_result, tuple):
                bw_result = (bw_result,)

            return bw_result

    _orig_params = orig_forward_params

    def aten_fwd(*args, **kwargs):
        user_inputs = _normalize_user_inputs(args, kwargs, _orig_params)
        all_inputs = _assemble_inputs(user_inputs, slot_specs, min_user_inputs)
        return _AtenGraph.apply(*all_inputs)

    return aten_fwd


def install(
    model: nn.Module,
    aten_module: types.ModuleType,
    *,
    validate: bool = True,
    param_paths: list[str] | None = None,
    num_real_outputs: int = 1,
    num_mutations: int = 0,
    mutated_buffers: list[str] | None = None,
    compile: bool = False,
) -> None:
    """Replace model's forward/backward with captured aten graphs.

    Drop-in replacement for torch.compile(model). After install(), model(x)
    runs the aten forward and loss.backward() runs the aten backward.
    Parameters are read directly so the optimizer sees them normally.

    When *compile* is True, the aten forward/backward are wrapped with
    ``torch.compile(backend="inductor")`` for Triton kernel fusion.
    """
    if param_paths is None:
        param_paths = _parse_param_paths(aten_module)

    forward_fn = build_aten_forward(
        model, aten_module,
        param_paths=param_paths,
        num_real_outputs=num_real_outputs,
        num_mutations=num_mutations,
        mutated_buffers=mutated_buffers,
        validate=validate,
        compile=compile,
    )

    model._original_forward = model.forward
    model.forward = forward_fn
    model._aten_installed = True
    model._aten_param_paths = param_paths
    model._aten_module = aten_module


def uninstall(model: nn.Module) -> None:
    """Restore the original forward method."""
    if hasattr(model, '_original_forward'):
        model.forward = model._original_forward
        del model._original_forward
        model._aten_installed = False


# -----------------------------------------------------------------------------
# Optimizer install
# -----------------------------------------------------------------------------


def install_optimizer_step(
    target_fn: Callable,
    aten_fn: Callable,
    *,
    validate_shapes: bool = True,
) -> Callable:
    """Replace a @torch.compile'd optimizer step with an aten version.

    Returns a wrapper that calls the aten version. Caller is responsible
    for monkey-patching it into the right module, e.g.:
        import nanochat.optim
        nanochat.optim.adamw_step_fused = install_optimizer_step(
            adamw_step_fused, opt_aten.adamw_step
        )
    """
    def wrapper(*args, **kwargs):
        return aten_fn(*args, **kwargs)

    wrapper._original = target_fn
    wrapper._aten_fn = aten_fn
    return wrapper


def capture_and_install(
    model: nn.Module,
    sample_args: tuple,
    sample_kwargs: dict | None = None,
    *,
    loss_fn: Callable | None = None,
    validate: bool = True,
) -> None:
    """One-shot: capture aten graphs and install them immediately (for round-trip testing)."""
    from torch_graph.export import capture_aten_graphs

    output, capture = capture_aten_graphs(
        model, *sample_args, run_backward=True, loss_fn=loss_fn, **(sample_kwargs or {}),
    )
    if not capture.forward_graphs:
        raise RuntimeError("No forward graphs captured")

    from torch_graph.auto_install import _compute_num_mutations, _find_mutated_buffer_paths

    fw_gm = capture.forward_graphs[0].graph_module
    bw_gm = capture.backward_graphs[0].graph_module if capture.backward_graphs else None
    param_paths = [n for n in capture.primal_names if n is not None]

    def aten_forward(*args): return fw_gm(*args)
    def aten_backward(*args):
        if bw_gm is None: raise RuntimeError("No backward graph captured")
        return bw_gm(*args)

    n_mutations = _compute_num_mutations(fw_gm, bw_gm)
    mutated_buffers = _find_mutated_buffer_paths(model, capture.primal_names, n_mutations)

    mod = types.SimpleNamespace(forward=aten_forward, backward=aten_backward)
    install(model, mod, validate=False, param_paths=param_paths,
            num_mutations=n_mutations, mutated_buffers=mutated_buffers)
