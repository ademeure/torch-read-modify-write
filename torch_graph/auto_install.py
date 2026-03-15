"""Drop-in torch.compile replacement: capture, edit, and reload aten graphs.

Import this module BEFORE any torch.compile calls to intercept them:

    import torch_graph.auto_install          # patches torch.compile
    torch_graph.auto_install.configure(...)   # optional settings

    # Now any script that calls torch.compile will use captured aten graphs
    # instead of Dynamo+Inductor.
    import my_training_script

How it works:
    1. torch.compile(model) is intercepted → first forward call triggers
       aten graph capture via aot_autograd
    2. The captured forward+backward graphs are saved as loadable .py files
    3. If a user-edited .py file already exists on disk, THAT is loaded instead
       (this is where you put custom CUDA/Triton kernels etc.)
    4. The model's forward is replaced with an autograd.Function that calls
       the captured/loaded forward+backward
    5. Same for @torch.compile'd functions (optimizer steps, etc.)

The on-disk .py module just needs to expose:
    forward(*args) -> tuple    # returns (real_outputs..., *saved_for_backward)
    backward(*args) -> tuple   # takes (*saved, *grad_outputs), returns grads

These can contain ANY code — triton kernels, custom CUDA, numpy, whatever.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import re
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn

logger = logging.getLogger("torch_graph")


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class AutoInstallConfig:
    """Global configuration for auto_install."""

    # Where to store/load cached aten files
    cache_dir: str = ".torch_graph_cache"

    # If True, always re-capture even if cache exists (but still prefers
    # user-modified files over freshly captured ones)
    force_recapture: bool = False

    # If True, validate shapes at install time and raise on mismatch
    validate_shapes: bool = True

    # If True, print status messages during capture/install
    verbose: bool = True

    # If True, run backward capture (needed for training)
    capture_backward: bool = True

    # Loss function for backward capture (None = output.sum())
    loss_fn: Callable | None = None

    # Number of real outputs from forward (vs saved tensors for backward)
    num_real_outputs: int = 1

    # Dynamic shapes (default True — batch dims etc. should not be hardcoded)
    dynamic: bool = True

    # If True, record real tensor values in captured aten files.
    # True: standalone .py files have real weight values inline.
    # False (default): saves memory, weights come from model at runtime.
    record_real_tensors: bool = False

    # If True, generate HTML graph visualization alongside each captured aten file
    generate_graph: bool = False

    # If True, dump H5 tensor files alongside each captured aten file
    dump_h5: bool = False

    # If True, also dump H5 for compiled functions (optimizer steps etc.)
    # Only has effect when dump_h5 is True.
    dump_h5_functions: bool = False

    # If True, skip saving .pt tensor files alongside aten .py files.
    # Useful when dump_h5 is True (H5 already has the tensors).
    skip_pt: bool = False

    # Exit after the captured aten graph has been called this many times
    # (0 = never exit).  The count starts AFTER capture — i.e. the first
    # real forward through the installed aten code counts as call 1.
    # Set to 1 to exit right after the first real call (the training step
    # whose forward triggered capture).  Set to a few to let the optimizer
    # also get captured (optimizers are @torch.compile'd and capture on
    # their first call, which happens a few calls into the training loop).
    # Useful for scripts with no __name__ guard (autoresearch, nanoGPT).
    exit_after_capture: int = 0

    # Shrink the batch dimension of tensor args to this size during capture.
    # 0 = no shrinking (use args as-is).  When > 0, all tensor args/kwargs
    # whose first dimension matches the detected batch size are sliced to
    # this size.  Requires dynamic=True so the captured graph generalizes.
    # Saves massive memory for large-batch scripts (e.g. autoresearch 128×2048).
    capture_batch_size: int = 0

    # Use inductor as the execution backend during capture.  The aten graphs
    # are still captured (for export), but the actual forward/backward runs
    # through inductor — giving the same memory efficiency as normal
    # torch.compile.  Without this, capture uses aot_eager which can OOM on
    # models that rely on inductor's op fusion to fit in memory.
    use_inductor: bool = False

    # Offload activations saved for backward to CPU during capture.
    # Dramatically reduces GPU memory for large models — only intermediate
    # activations are moved, NOT parameters or user inputs.  Slower due to
    # CPU↔GPU transfers, but allows capturing models that otherwise OOM.
    offload_saved: bool = False

    # Auto-capture optimizer.step() via patching torch.optim.Optimizer.step.
    # On first optimizer.step() call, captures aten graph and saves to disk.
    # Set False to skip optimizer capture entirely.
    capture_optimizer: bool = True

    # Replay captured optimizer aten on subsequent step() calls.
    # When True, steps 2+ run through the captured aten graph instead of eager.
    # When False (default), steps 2+ run eager (capture is for inspection only).
    replay_optimizer: bool = False

    # Run this many training steps then print loss summary and exit.
    # Implies replay_optimizer=True.  0 = disabled.
    verify_steps: int = 0

    # Record per-step losses to {cache_dir}/training_summary.json.
    record_steps: bool = False

    # Save a lossless JSON IR file (.json) alongside each captured aten .py file.
    # The JSON IR encodes the full FX graph (ops, shapes, dtypes, source locations)
    # in a structured format suitable for programmatic analysis or visualization.
    save_json_ir: bool = False


_config = AutoInstallConfig()
_call_count = 0  # how many times a captured model has been called
_step_losses: list[float] = []  # per-step loss values for verify/record


def configure(**kwargs) -> None:
    """Update auto_install configuration."""
    for k, v in kwargs.items():
        if not hasattr(_config, k):
            raise ValueError(f"Unknown config key: {k}")
        setattr(_config, k, v)


def get_config() -> AutoInstallConfig:
    """Get current configuration (read-only view)."""
    return _config


def _record_loss(result) -> None:
    """Extract scalar loss from forward result and append to _step_losses."""
    if not (_config.verify_steps > 0 or _config.record_steps):
        return
    try:
        if isinstance(result, torch.Tensor) and result.numel() == 1:
            _step_losses.append(result.item())
        elif isinstance(result, (tuple, list)) and len(result) > 0:
            first = result[0]
            if isinstance(first, torch.Tensor) and first.numel() == 1:
                _step_losses.append(first.item())
    except Exception:
        pass


def _print_verification_summary() -> None:
    """Print step/loss/delta table and optionally save to disk."""
    if not _step_losses:
        return
    print(f"\n{'='*50}")
    print(f"  Training verification: {len(_step_losses)} steps")
    print(f"{'='*50}")
    print(f"  {'Step':>4}  {'Loss':>12}  {'Delta':>12}")
    print(f"  {'----':>4}  {'----':>12}  {'-----':>12}")
    for i, loss in enumerate(_step_losses):
        delta = f"{loss - _step_losses[i-1]:+.6f}" if i > 0 else ""
        print(f"  {i+1:>4}  {loss:>12.6f}  {delta:>12}")
    print(f"{'='*50}\n")

    if _config.record_steps:
        summary_path = Path(_config.cache_dir) / "training_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump({"losses": _step_losses}, f, indent=2)
        logger.info(f"Training summary saved to {summary_path}")


# -----------------------------------------------------------------------------
# Mutation detection
# -----------------------------------------------------------------------------

def _compute_num_mutations(fg_gm, bg_gm) -> int:
    """Detect how many forward outputs are functionalized buffer mutations.

    aot_autograd converts in-place buffer updates (e.g. BatchNorm's
    running_mean += ...) into explicit forward return values.  The forward
    output tuple layout is:

        (mutation_0, ..., mutation_N, real_out_0, ..., saved_for_bw_0, ...)

    We detect N using: n_fw_outputs - n_tangents - n_bw_saved.

    If this count is wrong, models with BatchNorm/running stats will
    silently produce incorrect results (buffers never get updated).
    """
    if bg_gm is None:
        return 0
    fw_out_node = next(n for n in fg_gm.graph.nodes if n.op == 'output')
    fw_outputs = fw_out_node.args[0]
    n_fw = len(fw_outputs) if isinstance(fw_outputs, (tuple, list)) else 1

    bw_phs = [n for n in bg_gm.graph.nodes if n.op == 'placeholder']
    n_tangents = sum(1 for p in bw_phs if 'tangent' in p.name)
    n_bw_saved = len(bw_phs) - n_tangents

    return max(0, n_fw - n_tangents - n_bw_saved)


def _find_mutated_buffer_paths(model: nn.Module, primal_names: list, n_mutations: int) -> list[str]:
    """Identify which buffer paths correspond to the mutation outputs.

    aot_autograd orders mutation outputs to match the mutated inputs.
    We find buffers in primal_names order and return the first n_mutations.
    """
    if n_mutations == 0:
        return []
    buffer_set = set(dict(model.named_buffers()).keys())
    buffer_paths = []
    for name in primal_names:
        if name in buffer_set:
            buffer_paths.append(name)
        if len(buffer_paths) == n_mutations:
            break
    return buffer_paths


# -----------------------------------------------------------------------------
# Internal state
# -----------------------------------------------------------------------------

# Track all installed models/functions for introspection
_installed: dict[int, _InstalledEntry] = {}  # id(obj) → entry

# The real torch.compile, saved before patching
_real_torch_compile: Callable | None = None


@dataclass
class _InstalledEntry:
    """Tracks an installed aten replacement."""
    name: str
    kind: str  # "model" or "function"
    cache_path: str | None = None
    source: str = ""  # "captured", "loaded_from_disk", "wrapped"
    param_paths: list[str] = field(default_factory=list)


@dataclass
class _FnVariantEntry:
    """Installed state for one standalone-function variant."""
    forward: Callable
    source: str
    cache_path: str | None = None
    num_mutations: int = 0
    arg_to_placeholder: list[int | None] = field(default_factory=list)
    mutated_arg_indices: list[int] = field(default_factory=list)
    original_returns: bool = False
    symint_specs: list[tuple[int, int]] = field(default_factory=list)
    # Precomputed call-order mapping: [(call_idx, ...)] sorted by placeholder index.
    # Built lazily on first call to avoid recomputing the sort every invocation.
    _call_order: list[int] | None = field(default=None, repr=False)

    def get_call_order(self) -> list[int]:
        """Return call indices sorted by placeholder position (cached)."""
        if self._call_order is None:
            self._call_order = [
                call_idx
                for _, call_idx in sorted(
                    ((ph_idx, call_idx)
                     for call_idx, ph_idx in enumerate(self.arg_to_placeholder)
                     if ph_idx is not None),
                    key=lambda x: x[0],
                )
            ]
        return self._call_order


# -----------------------------------------------------------------------------
# Cache naming
# -----------------------------------------------------------------------------

def _model_cache_key(model: nn.Module) -> str:
    """Stable cache key from class name + param names/shapes/dtypes.

    Changing the model architecture (adding layers, changing shapes) produces
    a new cache key, triggering re-capture.
    """
    parts = [type(model).__qualname__]
    for name, p in model.named_parameters():
        parts.append(f"{name}:{list(p.shape)}:{p.dtype}")
    for name, b in model.named_buffers():
        parts.append(f"{name}:{list(b.shape)}:{b.dtype}")
    key_str = "|".join(parts)
    h = hashlib.sha256(key_str.encode()).hexdigest()[:12]
    return f"{type(model).__name__}_{h}"


def _fn_cache_key(fn: Callable) -> str:
    """Generate a cache key for a standalone compiled function."""
    name = getattr(fn, '__qualname__', getattr(fn, '__name__', 'fn'))
    # Include source location for uniqueness
    try:
        src_file = inspect.getfile(fn)
        src_lines, start_line = inspect.getsourcelines(fn)
        src_hash = hashlib.sha256("".join(src_lines).encode()).hexdigest()[:12]
    except (TypeError, OSError):
        src_hash = hashlib.sha256(name.encode()).hexdigest()[:12]
    safe_name = re.sub(r'[^\w]', '_', name)
    return f"{safe_name}_{src_hash}"


def _cache_path(key: str, variant: str | None = None) -> Path:
    """Full path for a cache entry, optionally with a variant suffix."""
    if variant:
        return Path(_config.cache_dir) / f"{key}_{variant}_aten.py"
    return Path(_config.cache_dir) / f"{key}_aten.py"


def _variant_key(args: tuple, kwargs: dict, training: bool) -> tuple:
    """Compute a lightweight variant key for O(1) dispatch.

    Each distinct combination of (n_args, non-None kwargs, train/eval mode)
    maps to a separate aten graph capture.
    """
    n_args = len(args)
    non_none_kw = frozenset(
        k for k, v in kwargs.items() if v is not None
    ) if kwargs else frozenset()
    return (n_args, non_none_kw, training)


def _shrink_batch(args: tuple, kwargs: dict, target_batch: int):
    """Shrink tensor args/kwargs along the batch dimension (dim 0).

    Detects the most common dim-0 size among tensors and slices any tensor
    with that size down to ``target_batch``.  Returns new (args, kwargs).
    """
    # Collect all dim-0 sizes from tensors
    from collections import Counter
    dim0_counts: Counter = Counter()
    for a in args:
        if isinstance(a, torch.Tensor) and a.dim() >= 1:
            dim0_counts[a.shape[0]] += 1
    for v in kwargs.values():
        if isinstance(v, torch.Tensor) and v.dim() >= 1:
            dim0_counts[v.shape[0]] += 1

    if not dim0_counts:
        return args, kwargs

    # The batch size is the most common dim-0 value
    batch_size = dim0_counts.most_common(1)[0][0]
    if batch_size <= target_batch:
        return args, kwargs

    logger.info(
        f"Shrinking batch {batch_size} → {target_batch} for capture "
        f"(saves ~{(batch_size - target_batch) / batch_size * 100:.0f}% activation memory)"
    )

    def _slice(t):
        if isinstance(t, torch.Tensor) and t.dim() >= 1 and t.shape[0] == batch_size:
            return t[:target_batch].contiguous()
        return t

    new_args = tuple(_slice(a) for a in args)
    new_kwargs = {k: _slice(v) for k, v in kwargs.items()}
    return new_args, new_kwargs


def _variant_suffix(key: tuple) -> str:
    """Human-readable suffix for variant cache file naming.

    Examples: '2a_train', '1a_eval', '1a_kv_cache_eval'
    """
    n_args, non_none_kw, training = key
    parts = [f"{n_args}a"]
    if non_none_kw:
        parts.extend(sorted(non_none_kw))
    parts.append("train" if training else "eval")
    return "_".join(parts)


def _fn_variant_arg_spec(arg: Any, dynamic: bool) -> tuple:
    """Build a hashable dispatch spec for a standalone compiled function arg."""
    if isinstance(arg, torch.Tensor):
        spec = (
            "tensor",
            str(arg.dtype),
            str(arg.device),
            arg.dim(),
        )
        if not dynamic:
            spec += (tuple(arg.shape),)
        return spec
    if isinstance(arg, (int, float, bool, str, type(None))):
        return ("value", type(arg).__qualname__, arg)
    if isinstance(arg, (tuple, list)):
        return (
            type(arg).__qualname__,
            tuple(_fn_variant_arg_spec(v, dynamic) for v in arg),
        )
    return ("object", type(arg).__qualname__, repr(arg))


def _fn_variant_key(args: tuple, kwargs: dict, dynamic: bool) -> tuple:
    """Dispatch key for a standalone compiled function.

    When ``dynamic=False``, tensor shapes participate in the key so each static
    shape pattern gets its own cached aten graph. Non-tensor args are treated
    as baked-in constants and always participate in the key.
    """
    return (
        dynamic,
        tuple(_fn_variant_arg_spec(arg, dynamic) for arg in args),
        tuple(
            sorted(
                (name, _fn_variant_arg_spec(value, dynamic))
                for name, value in (kwargs or {}).items()
            )
        ),
    )


def _fn_variant_suffix(key: tuple) -> str:
    """Stable cache suffix for a standalone function variant."""
    dynamic, arg_specs, _ = key
    digest = hashlib.sha256(repr(key).encode()).hexdigest()[:10]
    mode = "dyn" if dynamic else "static"
    return f"{len(arg_specs)}a_{mode}_{digest}"


# -----------------------------------------------------------------------------
# Loading aten modules from disk
# -----------------------------------------------------------------------------

def _load_aten_module(path: Path) -> types.ModuleType | None:
    """Load an aten .py file as a module — always reads the current file content.

    Evicts any previously cached version from sys.modules so that user edits
    to the .py file are picked up without needing to clear __pycache__.

    During loading, real torch.compile is restored so user-edited aten files
    can use torch.compile (e.g. to fuse custom kernels).
    """
    module_name = f"_torch_graph_aten_{path.stem}"
    # Evict stale version so the file is always re-read from disk.
    # Execute source directly instead of going through SourceFileLoader, which
    # may reuse a same-second .pyc if the edit preserved file size.
    sys.modules.pop(module_name, None)
    mod = types.ModuleType(module_name)
    mod.__file__ = str(path)
    mod.__package__ = ""
    mod.__builtins__ = __builtins__
    sys.modules[module_name] = mod
    source = path.read_text()
    try:
        code = compile(source, str(path), "exec")
    except (SyntaxError, ValueError) as e:
        logger.warning(
            "Corrupt or truncated aten cache file %s: %s. "
            "Will re-capture.", path, e
        )
        sys.modules.pop(module_name, None)
        return None
    # Restore real torch.compile so aten files can use it
    _unpatch_for_capture()
    try:
        exec(code, mod.__dict__)
    except ModuleNotFoundError as e:
        # Give actionable error for missing Triton when loading aten files
        # that contain Triton kernel definitions
        if "triton" in str(e) and "import triton" in source:
            raise ModuleNotFoundError(
                f"Aten file {path.name} contains Triton kernels but 'triton' is not installed. "
                f"Install with: pip install triton"
            ) from e
        raise
    finally:
        _repatch_after_capture()
    return mod


def _file_hash(path: Path) -> str:
    """SHA-256 of a file's content (first 12 hex chars)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _has_user_modified(path: Path) -> bool:
    """True if .py content differs from what was originally captured."""
    meta = _read_meta(path)
    if not meta:
        # No meta = user created this file manually → treat as user-modified
        return True
    original_hash = meta.get("file_hash")
    if original_hash is None:
        return False
    return _file_hash(path) != original_hash


def _read_meta(path: Path) -> dict:
    """Read metadata for a cached aten file, returning {} if missing or corrupt."""
    meta_path = path.with_suffix('.meta')
    if not meta_path.exists():
        return {}
    try:
        with open(meta_path) as f:
            return json.load(f)
    except Exception as e:
        logger.debug("Failed to read meta %s: %s", meta_path, e)
        return {}


def _write_meta(path: Path, info: dict) -> None:
    """Write metadata alongside a cached aten file."""
    # Store hash of the .py file so we can detect user edits by content, not mtime
    if path.exists():
        info["file_hash"] = _file_hash(path)
    meta_path = path.with_suffix('.meta')
    with open(meta_path, 'w') as f:
        json.dump(info, f, indent=2)


# -----------------------------------------------------------------------------
# Shape validation
# -----------------------------------------------------------------------------

def _save_capture_json_ir(capture, cache_path: Path) -> None:
    """Save a lossless JSON IR file alongside a captured aten .py file."""
    try:
        from torch_graph.ir_json import save_ir_json
        json_path = cache_path.with_suffix('.json')
        save_ir_json(capture, str(json_path))
        if _config.verbose:
            logger.info(f"Saved JSON IR: {json_path}")
    except Exception as e:
        logger.debug(f"JSON IR save failed for {cache_path}: {e}")


def _save_capture_artifacts(
    capture, cache_path: Path, variant: str,
    *, require_h5_functions: bool = False,
) -> None:
    """Save optional artifacts (JSON IR, graph visualization, H5 dump).

    Parameters
    ----------
    capture : CaptureResult
        The capture object produced by ``capture_aten_graphs``.
    cache_path : Path
        Path to the ``.py`` aten file on disk.
    variant : str
        Human-readable variant label (e.g. ``"train"``, ``"optimizer"``).
    require_h5_functions : bool
        When *True*, H5 dumps are only written if **both**
        ``_config.dump_h5`` and ``_config.dump_h5_functions`` are enabled.
        Model captures pass *False* (they only check ``dump_h5``); function
        and optimizer captures pass *True*.
    """
    if _config.save_json_ir:
        _save_capture_json_ir(capture, cache_path)
    if _config.generate_graph:
        _generate_graph(capture, cache_path, variant)
    should_dump_h5 = _config.dump_h5 and (
        not require_h5_functions or _config.dump_h5_functions
    )
    if should_dump_h5:
        _dump_h5(capture, cache_path, variant)


def _generate_graph(capture, cache_path: Path, variant: str) -> None:
    """Generate HTML graph visualization for a captured variant."""
    try:
        from torch_graph.visualizer import GraphVisualizer

        fw_gm = capture.forward_graphs[0].graph_module
        bw_gm = (capture.backward_graphs[0].graph_module
                 if capture.backward_graphs else None)

        stem = cache_path.stem  # e.g. GPT_abc_2a_train_aten
        html_path = cache_path.parent / f"{stem}.html"
        viz = GraphVisualizer(fw_gm)
        viz.save_html(
            str(html_path),
            title=f"aten — {variant}",
            source_map=capture.source_map,
            backward_source=bw_gm,
        )
        logger.info(f"Saved graph: {html_path}")
    except Exception as e:
        logger.warning(f"Graph generation failed for variant '{variant}': {e}")


def _dump_h5(capture, cache_path: Path, variant: str) -> None:
    """Dump H5 tensor files for a captured variant."""
    try:
        from torch_graph.op_dump import dump_grouped_tensors

        h5_path = cache_path.parent / f"{cache_path.stem}.h5"
        dump_grouped_tensors(
            capture,
            str(h5_path),
            group_by=["line", "module"],
            which="both",
            include_params=True,
            stats=True,
            replay_scripts=True,
        )
        logger.info(f"Saved H5 dump: {h5_path}")
    except Exception as e:
        logger.warning(f"H5 dump failed for variant '{variant}': {e}")


def _validate_model_shapes(
    model: nn.Module,
    aten_mod: types.ModuleType,
    param_paths: list[str],
    label: str = "",
) -> None:
    """Validate aten module's expected shapes match the model's parameters."""
    from torch_graph.install import _validate_shapes
    _validate_shapes(model, param_paths, aten_mod.forward, label=label or type(model).__name__)


def _try_load_from_cache(
    cache_path: Path,
    loader_fn: Callable,
    label: str,
    *,
    legacy_path: Path | None = None,
    has_variants: bool = True,
    recover_on_error: bool = True,
) -> object | None:
    """Shared cache-loading logic for both model and function proxies.

    Parameters
    ----------
    cache_path : Path
        Primary cache file to check.
    loader_fn : callable
        ``loader_fn(path)`` → loaded result, or raises on failure.
    label : str
        Human-readable label for log messages (e.g. model class name).
    legacy_path : Path | None
        Optional legacy (no-variant-suffix) path for backward compat.
    has_variants : bool
        If *True*, skip the legacy path fallback (caller already has loaded
        variants, so legacy files are irrelevant).
    recover_on_error : bool
        If *True*, wrap the non-user-modified primary load in try/except so a
        corrupt cache file falls through to recapture (model proxy behavior).
        If *False*, let exceptions propagate from the primary path (function
        proxy behavior).

    Returns
    -------
    The loaded result on cache hit, or *None* on cache miss (needs capture).
    """
    result = None

    if cache_path.exists():
        user_mod = _has_user_modified(cache_path)
        if user_mod:
            # User-modified → always load, let errors propagate
            result = loader_fn(cache_path)
        elif not _config.force_recapture:
            if recover_on_error:
                try:
                    result = loader_fn(cache_path)
                except Exception as e:
                    logger.warning(
                        "Failed to load cached aten file %s: %s. Re-capturing.",
                        cache_path, e,
                    )
                    result = None
            else:
                result = loader_fn(cache_path)

    # Legacy fallback — only when no variants have been loaded yet
    if result is None and legacy_path is not None and not has_variants:
        if legacy_path.exists():
            if _has_user_modified(legacy_path) or not _config.force_recapture:
                try:
                    result = loader_fn(legacy_path)
                except Exception as e:
                    logger.debug(
                        "Legacy cache file %s incompatible: %s", legacy_path, e,
                    )

    return result


def _serialize_input_shapes(args: tuple, kwargs: dict) -> list[dict]:
    """Serialize input arg shapes/dtypes for metadata storage and comparison."""
    shapes = []
    for a in args:
        if isinstance(a, torch.Tensor):
            shapes.append({"shape": list(a.shape), "dtype": str(a.dtype)})
        else:
            shapes.append({"type": type(a).__name__})
    for k, v in kwargs.items():
        if isinstance(v, torch.Tensor):
            shapes.append({"name": k, "shape": list(v.shape), "dtype": str(v.dtype)})
    return shapes


# -----------------------------------------------------------------------------
# Model install (via autograd.Function)
# -----------------------------------------------------------------------------

def _install_model_from_module(
    model: nn.Module,
    aten_mod: types.ModuleType,
    param_paths: list[str],
    *,
    num_real_outputs: int = 1,
    num_mutations: int = 0,
    mutated_buffers: list[str] | None = None,
) -> None:
    """Wrap aten module in autograd.Function and replace model.forward."""
    from torch_graph.install import install
    install(
        model, aten_mod,
        validate=False,  # already validated
        param_paths=param_paths,
        num_real_outputs=num_real_outputs,
        num_mutations=num_mutations,
        mutated_buffers=mutated_buffers or [],
    )


_capture_depth = 0  # re-entrant counter for nested capture contexts


def _unpatch_for_capture():
    """Temporarily restore real torch.compile during capture.

    Re-entrant: nested calls increment a depth counter; torch.compile
    is only restored on the outermost call and re-patched when depth
    returns to 0.  While depth > 0 any _CompiledFnProxy.__call__
    passes through to the original unwrapped function so that
    @torch.compile-decorated helpers (e.g. nanochat's adamw_step_fused)
    are traced normally by the outer dynamo instead of triggering a
    nested capture.
    """
    global _capture_depth
    _capture_depth += 1
    if _capture_depth == 1 and _real_torch_compile is not None:
        torch.compile = _real_torch_compile


def _repatch_after_capture():
    """Re-install our patched torch.compile after capture."""
    global _capture_depth
    _capture_depth -= 1
    if _capture_depth == 0 and _real_torch_compile is not None:
        torch.compile = _patched_compile


# -----------------------------------------------------------------------------
# Patched torch.compile for nn.Module
# -----------------------------------------------------------------------------

class _CompiledModelProxy:
    """Multi-variant dispatcher: each distinct call pattern gets its own aten graph.

    torch.compile(model) returns this proxy. Each unique call pattern
    (different arg count, non-None kwargs, train/eval mode) triggers
    capture of a separate aten graph, saved as an independently editable
    .py file. Variant dispatch is O(1) dict lookup per call.

    All attribute access is forwarded to the underlying model so training
    loops work transparently.
    """

    _PROXY_ATTRS = {
        '_model', '_compile_kwargs', '_cache_key',
        '_variants', '_orig_forward_params', '_inductor_compiled',
    }

    def __init__(self, model: nn.Module, compile_kwargs: dict):
        object.__setattr__(self, '_model', model)
        object.__setattr__(self, '_compile_kwargs', compile_kwargs)
        object.__setattr__(self, '_cache_key', _model_cache_key(model))
        object.__setattr__(self, '_variants', {})  # variant_key → callable
        from torch_graph.install import _extract_forward_arg_names
        object.__setattr__(self, '_orig_forward_params',
                          _extract_forward_arg_names(model.forward))
        object.__setattr__(self, '_inductor_compiled', None)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, '_model'), name)

    def __setattr__(self, name, value):
        if name in _CompiledModelProxy._PROXY_ATTRS:
            object.__setattr__(self, name, value)
        else:
            setattr(self._model, name, value)

    def __call__(self, *args, **kwargs):
        training = self._model.training
        key = _variant_key(args, kwargs, training)
        forward = self._variants.get(key)

        if forward is None:
            if not training:
                # ── Eval optimization ────────────────────────────────────
                # We always capture eval to disk so users CAN edit it, but
                # we DON'T install it unless they actually have edited it.
                # Running hundreds of decomposed aten ops individually is
                # ~2-5x slower than PyTorch's fused eager kernels.  So for
                # unmodified eval files, we skip the aten path entirely.
                suffix = _variant_suffix(key)
                variant_path = _cache_path(self._cache_key, suffix)
                user_modified = variant_path.exists() and _has_user_modified(variant_path)
                if not variant_path.exists() or (
                    _config.force_recapture and not user_modified
                ):
                    self._capture_variant(key, args, kwargs, variant_path,
                                          skip_load=True)
                    user_modified = False
                if user_modified:
                    forward = self._get_or_create_variant(key, args, kwargs)
                else:
                    self._variants[key] = None  # mark as seen
                    return self._model.forward(*args, **kwargs)
            else:
                forward = self._get_or_create_variant(key, args, kwargs)

        if forward is None:
            return self._model.forward(*args, **kwargs)

        result = forward(*args, **kwargs)
        _record_loss(result)

        if _config.exit_after_capture > 0:
            global _call_count
            _call_count += 1
            if _call_count >= _config.exit_after_capture:
                cache_dir = Path(_config.cache_dir)
                if _config.verify_steps > 0:
                    _print_verification_summary()
                logger.info(
                    f"exit_after_capture={_config.exit_after_capture} reached "
                    f"({_call_count} call(s)). Cache: {cache_dir}"
                )
                sys.exit(0)

        return result

    def _get_or_create_variant(self, key, args, kwargs):
        """Load from cache or capture a new variant."""
        suffix = _variant_suffix(key)
        variant_path = _cache_path(self._cache_key, suffix)

        forward = _try_load_from_cache(
            variant_path,
            self._load_variant,
            type(self._model).__name__,
            legacy_path=_cache_path(self._cache_key),
            has_variants=bool(self._variants),
            recover_on_error=True,
        )

        # Capture new variant if cache miss
        if forward is None:
            forward = self._capture_variant(key, args, kwargs, variant_path)

        self._variants[key] = forward

        n = len(self._variants)
        if _config.verbose:
            logger.info(
                f"{type(self._model).__name__}: variant '{suffix}' ready "
                f"({n} variant{'s' if n > 1 else ''} total)"
            )

        _installed[id(self._model)] = _InstalledEntry(
            name=type(self._model).__name__,
            kind="model",
            cache_path=str(variant_path),
            source=f"{n} variant(s)",
        )

        return forward

    def _load_variant(self, cache_path: Path) -> Callable | None:
        """Load an aten .py file and build a forward callable."""
        if _config.verbose:
            tag = " (user-modified)" if _has_user_modified(cache_path) else ""
            logger.info(f"Loading aten{tag}: {cache_path}")

        aten_mod = _load_aten_module(cache_path)
        if aten_mod is None:
            return None

        from torch_graph.install import _parse_param_paths, build_aten_forward
        param_paths = _parse_param_paths(aten_mod)
        if not param_paths:
            raise ValueError(
                f"Cannot determine parameter mapping from {cache_path}. "
                f"Ensure the file has a 'Parameter mapping:' section."
            )

        if _config.validate_shapes and not _config.dynamic:
            _validate_model_shapes(self._model, aten_mod, param_paths)

        num_real = _config.num_real_outputs
        if hasattr(aten_mod, 'NUM_REAL_OUTPUTS'):
            num_real = aten_mod.NUM_REAL_OUTPUTS

        meta = _read_meta(cache_path)
        n_mutations = meta.get("num_mutations", 0)
        mutated_buffers = meta.get("mutated_buffers", [])
        user_input_order = meta.get("user_input_order")

        return build_aten_forward(
            self._model, aten_mod,
            param_paths=param_paths,
            orig_forward_params=self._orig_forward_params,
            num_real_outputs=num_real,
            num_mutations=n_mutations,
            mutated_buffers=mutated_buffers,
            validate=False,
            user_input_order=user_input_order,
        )

    def _capture_variant(self, key, args, kwargs, cache_path: Path, *,
                          skip_load: bool = False) -> Callable | None:
        """Capture aten graphs for a new call pattern and save to disk.

        Uses a deepcopy of the model for capture so the original model's
        parameters and RNG state are not disturbed.  RNG state is saved
        and restored because torch.compile + aot_autograd consume random
        state during tracing.

        If skip_load=True, only saves to disk without loading back.  Used
        by the eval optimization where we capture to disk for editability
        but skip the aten path unless the user has modified the file.
        """
        suffix = _variant_suffix(key)
        if _config.verbose:
            logger.info(
                f"Capturing aten for {type(self._model).__name__} "
                f"variant '{suffix}'..."
            )

        import copy
        from torch_graph.export import capture_aten_graphs, export_aten_program

        # Skip deepcopy when we're going to exit anyway — saves memory for
        # large models (autoresearch: 50M params + batch=128×2048).
        if _config.exit_after_capture > 0:
            capture_model = self._model
        else:
            capture_model = copy.deepcopy(self._model)
        saved_rng = torch.random.get_rng_state()
        saved_cuda_rng = (torch.cuda.get_rng_state_all()
                         if torch.cuda.is_available() else None)

        # Only capture backward for training-mode variants
        _, _, training = key
        run_bw = _config.capture_backward and training

        _unpatch_for_capture()
        # Respect the original torch.compile(dynamic=...) if specified,
        # otherwise fall back to the global config.
        use_dynamic = self._compile_kwargs.get('dynamic', _config.dynamic)

        # Shrink batch dimension to save memory during capture.
        capture_args = args
        capture_kwargs = kwargs
        if _config.capture_batch_size > 0:
            capture_args, capture_kwargs = _shrink_batch(
                args, kwargs, _config.capture_batch_size
            )
            # Force dynamic so the graph generalizes to any batch size
            use_dynamic = True

        try:
            output, capture = capture_aten_graphs(
                capture_model, *capture_args,
                run_backward=run_bw,
                loss_fn=_config.loss_fn,
                dynamic=use_dynamic,
                record_real_tensors=_config.record_real_tensors,
                use_inductor=_config.use_inductor,
                offload_saved=_config.offload_saved,
                **capture_kwargs,
            )
        finally:
            _repatch_after_capture()

        torch.random.set_rng_state(saved_rng)
        if saved_cuda_rng is not None:
            torch.cuda.set_rng_state_all(saved_cuda_rng)
        del capture_model

        if not capture.forward_graphs:
            raise RuntimeError(
                f"Failed to capture aten graphs for "
                f"{type(self._model).__name__} variant '{suffix}'"
            )

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Always skip .pt/.h5 weight files — auto_install uses live model
        # parameters, never the saved weights.  Also avoids SymInt issues
        # when capture_batch_size forces dynamic=True.
        export_aten_program(capture, str(cache_path), include_test_harness=False,
                            skip_pt=True)

        # Metadata — use original args shapes (not shrunk capture_args)
        input_shapes = _serialize_input_shapes(args, kwargs)

        fg_gm = capture.forward_graphs[0].graph_module
        bg_gm = (capture.backward_graphs[0].graph_module
                 if capture.backward_graphs else None)
        n_mutations = _compute_num_mutations(fg_gm, bg_gm)
        mutated_buffers = _find_mutated_buffer_paths(
            self._model, capture.primal_names, n_mutations
        )

        # Extract user input ordering from capture (aot_autograd may reorder).
        # Raw ordering has forward-signature positions; convert to normalized
        # positions (relative to the compacted output of _normalize_user_inputs,
        # which skips missing kwargs).
        user_input_order = None
        _uio_raw = getattr(capture, '_user_input_ordering', None)
        if _uio_raw:
            present = sorted(set(p for p in _uio_raw if p is not None))
            pos_to_idx = {pos: idx for idx, pos in enumerate(present)}
            user_input_order = [pos_to_idx[p] for p in _uio_raw if p is not None]

        # Serialize primal_names for standalone training loop generation.
        # None entries = user inputs, string entries = param/buffer paths.
        serializable_primal_names = list(capture.primal_names) if capture.primal_names else []

        _write_meta(cache_path, {
            "model": type(self._model).__qualname__,
            "cache_key": self._cache_key,
            "variant": suffix,
            "n_params": sum(p.numel() for p in self._model.parameters()),
            "num_real_outputs": _config.num_real_outputs,
            "num_mutations": n_mutations,
            "mutated_buffers": mutated_buffers,
            "input_shapes": input_shapes,
            "dynamic": use_dynamic,
            "user_input_order": user_input_order,
            "primal_names": serializable_primal_names,
        })

        if _config.verbose:
            logger.info(f"Saved aten cache: {cache_path}")

        _save_capture_artifacts(capture, cache_path, suffix)

        if skip_load:
            return None

        # When offloading is enabled and we're going to exit anyway, reuse
        # the aot_eager compiled model (which has offloading baked in) for
        # subsequent forward calls.  The installed aten model runs unfused
        # and would OOM on models that need inductor to fit in memory.
        if _config.offload_saved and _config.exit_after_capture > 0:
            compiled_model = getattr(capture, '_compiled', None)
            if compiled_model is not None:
                return compiled_model

        return self._load_variant(cache_path)


# -----------------------------------------------------------------------------
# Patched torch.compile for standalone functions
# -----------------------------------------------------------------------------

class _CompiledFnProxy:
    """Wraps a @torch.compile'd function (e.g. optimizer steps).

    On first call:
      1. Check for a user-provided/cached aten file on disk → load it
      2. Otherwise, auto-capture the function's aten graph via aot_autograd
      3. Fall back to eager if capture fails

    aot_autograd functionalizes in-place mutations: the captured FX graph
    is pure (no in-place ops) and returns the new values of mutated inputs.
    The wrapper detects which call-position args were mutated during capture
    and writes results back via copy_() on subsequent calls.

    Non-tensor args (int, float) are baked into the graph as constants.
    Unused inputs may be eliminated from the graph. The wrapper handles
    both cases by tracking the mapping from call-position args to FX
    placeholders.
    """

    def __init__(self, fn: Callable, compile_kwargs: dict):
        self._fn = fn
        self._compile_kwargs = compile_kwargs
        self._cache_key = _fn_cache_key(fn)
        self._variants: dict[tuple, _FnVariantEntry] = {}

    def __call__(self, *args, **kwargs):
        # Record this call if we're building an inner fn replay plan.
        # Matching happens NOW (before the inner fn executes) so stacked
        # tensors can be identified before in-place mutations scramble them.
        if _recording_inner_calls and _recording_optimizer is not None:
            arg_roles = []
            for a in args:
                role = _match_arg_to_optimizer(a, _recording_optimizer)
                # Snapshot optimizer_attr values — they may differ between
                # calls (e.g. muon LR prescaling fills _muon_lr_t with a
                # different value for each param group).
                if role["role"] == "optimizer_attr" and isinstance(a, torch.Tensor):
                    try:
                        role["captured_value"] = a.item()
                    except Exception:
                        role["captured_value"] = a.clone().detach()
                arg_roles.append(role)
            _inner_call_records.append({
                "proxy": self,
                "arg_roles": arg_roles,
            })

        # During capture (_capture_depth > 0), pass through to the original
        # function so dynamo can trace it normally instead of hitting our proxy.
        # This handles @torch.compile-decorated helpers (e.g. nanochat's
        # adamw_step_fused) that are called inside an outer capture context.
        if _capture_depth > 0:
            return self._fn(*args, **kwargs)

        key = self._dispatch_key(args, kwargs)
        variant = self._variants.get(key)
        if variant is None:
            return self._first_call(key, args, kwargs)
        self._mark_installed(variant)
        return self._call_variant(variant, args, kwargs)

    def _first_call(self, key, args, kwargs):
        """Resolve a new call variant: load from cache, capture, or fall back."""
        fn_name = self._fn_name()
        suffix = _fn_variant_suffix(key)
        cache_path = _cache_path(self._cache_key, suffix)

        variant = self._load_cached_variant(args, cache_path, fn_name)
        if variant is not None:
            self._variants[key] = variant
            self._mark_installed(variant)
            return self._call_variant(variant, args, kwargs)

        try:
            variant, output = self._capture_variant(
                key, args, kwargs, cache_path, fn_name
            )
            self._variants[key] = variant
            self._mark_installed(variant)
            return output
        except Exception as e:
            logger.warning(
                f"Failed to capture aten for {fn_name} variant '{suffix}': {e}. "
                "Falling back to eager."
            )

        variant = self._fallback_eager(fn_name)
        self._variants[key] = variant
        self._mark_installed(variant)
        return self._call_variant(variant, args, kwargs)

    def _dispatch_key(self, args, kwargs) -> tuple:
        return _fn_variant_key(args, kwargs, self._use_dynamic())

    def _use_dynamic(self) -> bool:
        return self._compile_kwargs.get('dynamic', _config.dynamic)

    def _fn_name(self) -> str:
        return getattr(self._fn, '__qualname__', str(self._fn))

    def _load_cached_variant(
        self,
        args,
        cache_path: Path,
        fn_name: str,
    ) -> _FnVariantEntry | None:
        return _try_load_from_cache(
            cache_path,
            lambda p: self._load_from_disk(p, fn_name, args),
            fn_name,
            legacy_path=_cache_path(self._cache_key),
            has_variants=bool(self._variants),
            recover_on_error=False,
        )

    def _load_from_disk(
        self,
        cache_path: Path,
        fn_name: str,
        args,
    ) -> _FnVariantEntry | None:
        """Load a cached/user-provided aten file and restore mutation metadata."""
        if _config.verbose:
            tag = " (user-modified)" if _has_user_modified(cache_path) else ""
            logger.info(f"Loading aten{tag} for {fn_name}: {cache_path}")
        aten_mod = _load_aten_module(cache_path)
        forward_fn = self._find_forward_in_module(aten_mod)
        if forward_fn is None:
            return None

        meta = _read_meta(cache_path)
        num_mutations = meta.get("num_mutations", 0)
        mutated_arg_indices = meta.get("mutated_arg_indices", [])
        arg_to_placeholder = meta.get("arg_to_placeholder", [])
        original_returns = meta.get("original_returns", False)
        symint_specs = [tuple(s) for s in meta.get("symint_specs", [])]

        if not arg_to_placeholder:
            ph = 0
            arg_to_placeholder = []
            for i, a in enumerate(args):
                if isinstance(a, torch.Tensor):
                    arg_to_placeholder.append(ph)
                    ph += 1
                else:
                    arg_to_placeholder.append(None)

        return _FnVariantEntry(
            forward=forward_fn,
            source="loaded_from_disk",
            cache_path=str(cache_path),
            num_mutations=num_mutations,
            arg_to_placeholder=arg_to_placeholder,
            mutated_arg_indices=mutated_arg_indices,
            original_returns=original_returns,
            symint_specs=symint_specs,
        )

    def _capture_variant(self, key, args, kwargs, cache_path: Path, fn_name: str):
        """Capture aten graphs for the function and save to disk.

        Returns the original function's return value (the capture runs the
        function eagerly, so mutations have already happened).
        """
        suffix = _fn_variant_suffix(key)
        if _config.verbose:
            logger.info(f"Capturing aten for {fn_name} variant '{suffix}'...")

        from torch_graph.export import capture_aten_graphs, export_aten_program

        unwrapped = self._fn
        if hasattr(unwrapped, '__wrapped__'):
            unwrapped = unwrapped.__wrapped__

        tensor_clones = {}
        for i, a in enumerate(args):
            if isinstance(a, torch.Tensor):
                tensor_clones[i] = a.clone()

        use_dynamic = self._use_dynamic()

        _unpatch_for_capture()
        try:
            output, capture = capture_aten_graphs(
                unwrapped, *args, run_backward=False,
                dynamic=use_dynamic,
                record_real_tensors=_config.record_real_tensors,
                **kwargs,
            )
        finally:
            _repatch_after_capture()

        if not capture.forward_graphs:
            raise RuntimeError(f"No aten graphs captured for {fn_name}")

        gm_for_count = capture.forward_graphs[0].graph_module
        output_node = next(
            n for n in gm_for_count.graph.nodes if n.op == 'output'
        )
        fx_output_count = (
            len(output_node.args[0])
            if isinstance(output_node.args[0], (tuple, list))
            else 1
        )

        original_returns = output is not None

        changed_indices = sorted([
            i for i, clone in tensor_clones.items()
            if not torch.equal(args[i], clone)
        ])

        if output is None:
            num_mutations = fx_output_count
        else:
            num_mutations = len(changed_indices)

        mutated_arg_indices = changed_indices

        gm = capture.forward_graphs[0].graph_module
        ph_nodes = [n for n in gm.graph.nodes if n.op == 'placeholder']

        fn_ordering = getattr(capture, '_fn_input_ordering', None)
        if fn_ordering is not None:
            arg_to_placeholder = [None] * len(args)
            for ph_idx, call_idx in enumerate(fn_ordering):
                if call_idx is not None and call_idx < len(args):
                    arg_to_placeholder[call_idx] = ph_idx
        else:
            n_symint_phs = sum(
                1 for n in ph_nodes
                if not isinstance(n.meta.get('val'), torch.Tensor)
            )
            ph = n_symint_phs
            arg_to_placeholder = []
            for i in range(len(args)):
                if isinstance(args[i], torch.Tensor):
                    arg_to_placeholder.append(ph)
                    ph += 1
                else:
                    arg_to_placeholder.append(None)

        symint_specs: list[tuple[int, int]] = []
        if fn_ordering is not None:
            for ph_idx, call_idx in enumerate(fn_ordering):
                if call_idx is not None:
                    continue
                ph_node = ph_nodes[ph_idx]
                try:
                    concrete = int(ph_node.meta.get('val'))
                except (TypeError, ValueError):
                    continue
                for next_ph_idx in range(ph_idx + 1, len(fn_ordering)):
                    next_call_idx = fn_ordering[next_ph_idx]
                    if next_call_idx is None:
                        continue
                    t_meta = ph_nodes[next_ph_idx].meta.get('val')
                    if not isinstance(t_meta, torch.Tensor):
                        continue
                    matched = False
                    for d, s in enumerate(t_meta.shape):
                        try:
                            if int(s) == concrete:
                                symint_specs.append((next_call_idx, d))
                                matched = True
                                break
                        except (TypeError, ValueError):
                            pass
                    if matched:
                        break

        if num_mutations > 0 and fn_ordering is not None:
            ph_pos = {
                call_idx: ph_idx
                for ph_idx, call_idx in enumerate(fn_ordering)
                if call_idx is not None
            }
            mutated_set = set(changed_indices)
            refined_indices: list[int] | None = None

            if len(mutated_set) < num_mutations:
                # Sentinel-based mutation detection: fill each tensor arg with
                # a unique large value, re-run, and check which outputs match.
                # Values start at 1000.0 and increment by 1000.0 to avoid
                # collisions with realistic tensor values.
                sentinel_inputs = []
                sentinel_vals: dict[int, float] = {}
                fill_val = 1000.0
                for ph_idx, call_idx in enumerate(fn_ordering):
                    if call_idx is None:
                        try:
                            sentinel_inputs.append(int(ph_nodes[ph_idx].meta.get('val')))
                        except (TypeError, ValueError):
                            sentinel_inputs.append(1)
                        continue

                    arg = args[call_idx]
                    if isinstance(arg, torch.Tensor) and arg.numel() > 1:
                        fill = fill_val
                        if not (arg.is_floating_point() or arg.is_complex()):
                            fill = int(fill_val)
                        sentinel_inputs.append(torch.full(
                            arg.shape,
                            fill,
                            dtype=arg.dtype,
                            device=arg.device,
                        ))
                        sentinel_vals[call_idx] = float(fill_val)
                        fill_val += 1000.0
                    elif isinstance(arg, torch.Tensor):
                        sentinel_inputs.append(arg.clone())
                    else:
                        sentinel_inputs.append(arg)

                try:
                    sentinel_result = gm_for_count(*sentinel_inputs)
                    if not isinstance(sentinel_result, (tuple, list)):
                        sentinel_result = (sentinel_result,)
                    available = dict(sentinel_vals)
                    refined = []
                    for out_idx in range(num_mutations):
                        out_t = sentinel_result[out_idx]
                        if not isinstance(out_t, torch.Tensor) or not available:
                            refined = []
                            break
                        out_mean = out_t.float().mean().item()
                        best_call_idx = min(
                            available,
                            key=lambda c: abs(out_mean - available[c]),
                        )
                        refined.append(best_call_idx)
                        del available[best_call_idx]
                    if len(refined) == num_mutations:
                        refined_indices = refined
                        mutated_set = set(refined)
                except Exception as e:
                    logger.debug(
                        "Sentinel mutation detection failed for %s: %s",
                        fn_name,
                        e,
                    )

            if refined_indices is None and len(mutated_set) < num_mutations:
                for _, call_idx in sorted(
                    ((p, c) for p, c in enumerate(fn_ordering) if c is not None),
                    key=lambda x: x[0],
                ):
                    if call_idx in mutated_set:
                        continue
                    arg = args[call_idx]
                    if not isinstance(arg, torch.Tensor) or arg.numel() <= 1:
                        continue
                    mutated_set.add(call_idx)
                    if len(mutated_set) == num_mutations:
                        break

            mutated_arg_indices = (
                refined_indices
                if refined_indices is not None
                else sorted(mutated_set, key=lambda c: ph_pos.get(c, 999))
            )

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        export_aten_program(capture, str(cache_path), include_test_harness=False,
                            skip_pt=_config.skip_pt)

        if _config.verbose:
            logger.info(f"Saved aten: {cache_path}")

        _write_meta(cache_path, {
            "function": fn_name,
            "cache_key": self._cache_key,
            "variant": suffix,
            "num_mutations": num_mutations,
            "mutated_arg_indices": mutated_arg_indices,
            "arg_to_placeholder": arg_to_placeholder,
            "original_returns": original_returns,
            "symint_specs": symint_specs,
            "dynamic": use_dynamic,
        })

        _save_capture_artifacts(capture, cache_path, suffix,
                                require_h5_functions=True)

        aten_mod = _load_aten_module(cache_path)
        forward_fn = self._find_forward_in_module(aten_mod)
        if forward_fn is None:
            raise RuntimeError(f"Exported aten file has no forward: {cache_path}")

        return _FnVariantEntry(
            forward=forward_fn,
            source="captured",
            cache_path=str(cache_path),
            num_mutations=num_mutations,
            arg_to_placeholder=arg_to_placeholder,
            mutated_arg_indices=mutated_arg_indices,
            original_returns=original_returns,
            symint_specs=symint_specs,
        ), output

    def _fallback_eager(self, fn_name: str) -> _FnVariantEntry:
        """Fall back to the unwrapped eager function for this variant."""
        unwrapped = self._fn
        if hasattr(unwrapped, '__wrapped__'):
            unwrapped = unwrapped.__wrapped__

        if _config.verbose:
            logger.info(f"Using eager (unwrapped) for {fn_name}")

        return _FnVariantEntry(
            forward=unwrapped,
            source="unwrapped",
        )

    def _mark_installed(self, variant: _FnVariantEntry) -> None:
        _installed[id(self._fn)] = _InstalledEntry(
            name=self._fn_name(),
            kind="function",
            cache_path=variant.cache_path,
            source=variant.source,
        )

    def _call_variant(self, variant: _FnVariantEntry, args, kwargs):
        """Call the resolved function.

        For captured aten: build FX inputs, call forward, write mutations back.
        For eager fallback: call directly (mutations happen in-place naturally).
        """
        if variant.source == "unwrapped":
            return variant.forward(*args, **kwargs)

        fx_inputs = []
        for call_idx, dim_idx in variant.symint_specs:
            fx_inputs.append(args[call_idx].shape[dim_idx])
        for call_idx in variant.get_call_order():
            fx_inputs.append(args[call_idx])

        result = variant.forward(*fx_inputs)

        if variant.num_mutations > 0:
            if isinstance(result, (tuple, list)):
                for out_idx, arg_idx in enumerate(variant.mutated_arg_indices):
                    args[arg_idx].copy_(result[out_idx])
                if variant.original_returns:
                    ret = result[variant.num_mutations:]
                    return ret[0] if len(ret) == 1 else ret
                return None
            args[variant.mutated_arg_indices[0]].copy_(result)
            return None

        if isinstance(result, (tuple, list)) and len(result) == 1:
            return result[0]
        return result

    @staticmethod
    def _find_forward_in_module(aten_mod) -> Callable | None:
        """Find the forward function in a loaded aten module."""
        if hasattr(aten_mod, 'forward'):
            return aten_mod.forward
        # Look for any public callable
        for attr_name in dir(aten_mod):
            if attr_name.startswith('_'):
                continue
            attr = getattr(aten_mod, attr_name)
            if callable(attr):
                return attr
        return None

    # Forward attribute access to original function
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        return getattr(self._fn, name)


# -----------------------------------------------------------------------------
# Optimizer replay
# -----------------------------------------------------------------------------


@dataclass
class _OptimizerReplayInfo:
    """Everything needed to replay an optimizer step through captured aten."""
    forward_fn: Callable                    # The captured aten forward
    slot_info: list[dict[str, Any]]         # Per-placeholder role mapping
    mutated_slot_indices: list[int]         # Which placeholder indices are mutated
    optimizer: Any                          # Weak ref not needed — optimizer owns us


@dataclass
class _InnerFnReplayPlan:
    """Replay plan for optimizers with inner @torch.compile'd functions.

    Instead of calling the original optimizer.step() on step 2+, we manage
    the outer loop ourselves: increment step counters, fill scalar tensors,
    stack/unstack params, and call each inner fn's captured aten graph.
    """
    optimizer: Any
    calls: list[dict]           # Each: {proxy, arg_roles, copy_back_groups}
    step_attr_names: list[str]  # optimizer attrs that hold step count values


# Recording state for building inner fn replay plans.
# Set during the first optimizer.step() for inner-compiled optimizers.
_recording_inner_calls: bool = False
_recording_optimizer: Any = None
_inner_call_records: list[dict] = []


def _detect_mutated_slots(
    slot_info: list[dict[str, Any]],
    num_outputs: int,
) -> list[int]:
    """Determine which FX placeholder indices are mutated.

    For optimizer.step() (returns None), ALL FX outputs are mutations.
    The mutations are written back to input placeholders in ascending
    placeholder order. Standard optimizers mutate params and state tensors
    (not grads). We trust the FX output count and select the first
    num_outputs param/state slots in placeholder order.
    """
    # Collect param and state slot indices (these are the mutable slots)
    mutable = [
        i for i, info in enumerate(slot_info)
        if info.get("role") in ("param", "state", "optimizer_attr")
    ]

    # If we have exactly the right number, use them
    if len(mutable) == num_outputs:
        return mutable

    # Fallback: assume the first num_outputs placeholders are mutated
    # (works for simple optimizers where params come first)
    return list(range(num_outputs))


def _enrich_unknown_slots(
    slot_info: list[dict[str, Any]],
    real_inputs: list,
    optimizer,
    input_ptrs: list[int | None] | None = None,
) -> None:
    """Re-match unknown slots against the now-populated optimizer state.

    After optimizer.step() runs, state tensors (exp_avg, exp_avg_sq, step)
    exist in optimizer.state. We match unknown slots using data_ptr from the
    live inputs captured during the FX compilation pass.
    """
    # Build a lookup from data_ptr → role info using the post-step state
    ptr_to_info: dict[int, dict[str, Any]] = {}
    for gi, group in enumerate(optimizer.param_groups):
        for gpi, param in enumerate(group["params"]):
            base = {
                "group_index": gi,
                "group_param_index": gpi,
                "param_name": f"group{gi}.param{gpi}",
            }
            try:
                ptr_to_info[param.data_ptr()] = {**base, "role": "param"}
            except Exception:
                pass
            if param.grad is not None:
                try:
                    ptr_to_info[param.grad.data_ptr()] = {**base, "role": "grad"}
                except Exception:
                    pass
            state = optimizer.state.get(param, {})
            for state_key, val in state.items():
                if isinstance(val, torch.Tensor):
                    try:
                        ptr_to_info[val.data_ptr()] = {
                            **base, "role": "state", "state_key": str(state_key),
                        }
                    except Exception:
                        pass

    # Also scan optimizer instance attributes for tensors (e.g. MuonAdamW
    # stores 0-D tensors like `rng_state` directly on the instance).
    for attr_name in dir(optimizer):
        if attr_name.startswith("_"):
            continue
        try:
            val = getattr(optimizer, attr_name)
        except Exception:
            continue
        if isinstance(val, torch.Tensor):
            try:
                ptr_to_info.setdefault(val.data_ptr(), {
                    "role": "optimizer_attr",
                    "attr_name": attr_name,
                })
            except Exception:
                pass

    if not input_ptrs:
        return

    # Match unknown slots by their original live data_ptr
    for i, info in enumerate(slot_info):
        if info.get("role") != "unknown":
            continue
        if i >= len(input_ptrs):
            continue
        ptr = input_ptrs[i]
        if ptr is None:
            continue
        match = ptr_to_info.get(ptr)
        if match:
            info.update(match)


def _build_optimizer_replay(
    optimizer,
    forward_fn: Callable,
    slot_info: list[dict[str, Any]],
    mutated_slot_indices: list[int],
) -> _OptimizerReplayInfo:
    """Build a replay info object from capture metadata.

    Each slot in slot_info maps to a live tensor accessor:
      - "param"  → optimizer.param_groups[gi]["params"][gpi]
      - "grad"   → same param's .grad
      - "state"  → optimizer.state[param][state_key]
      - "unknown" → frozen value from capture (logged as warning)
    """
    return _OptimizerReplayInfo(
        forward_fn=forward_fn,
        slot_info=slot_info,
        mutated_slot_indices=mutated_slot_indices,
        optimizer=optimizer,
    )


def _load_optimizer_replay(
    optimizer,
    cache_path: Path,
) -> _OptimizerReplayInfo | None:
    """Load optimizer replay info from a cached aten .py + .meta file."""
    meta = _read_meta(cache_path)
    slot_info = meta.get("slot_info")
    mutated_slot_indices = meta.get("mutated_slot_indices")

    if not slot_info or mutated_slot_indices is None:
        logger.debug(f"No replay metadata in {cache_path}.meta")
        return None

    # Load the aten module
    try:
        aten_mod = _load_aten_module(cache_path)
    except Exception as e:
        logger.warning(f"Failed to load optimizer aten from {cache_path}: {e}")
        return None

    forward_fn = getattr(aten_mod, "forward", None)
    if forward_fn is None:
        logger.warning(f"No forward() in optimizer aten module: {cache_path}")
        return None

    if _config.verbose:
        logger.info(
            f"Loaded optimizer replay from cache: {len(slot_info)} slots, "
            f"{len(mutated_slot_indices)} mutations"
        )

    return _build_optimizer_replay(
        optimizer, forward_fn, slot_info, mutated_slot_indices,
    )


def _run_optimizer_replay(replay: _OptimizerReplayInfo) -> None:
    """Execute one optimizer step through the captured aten graph."""
    optimizer = replay.optimizer

    # Assemble FX inputs from live optimizer state
    fx_inputs = []
    for info in replay.slot_info:
        role = info.get("role", "unknown")
        gi = info.get("group_index")
        gpi = info.get("group_param_index")

        if role == "param":
            fx_inputs.append(optimizer.param_groups[gi]["params"][gpi])
        elif role == "grad":
            param = optimizer.param_groups[gi]["params"][gpi]
            if param.grad is None:
                raise RuntimeError(
                    f"Optimizer replay: param group{gi}.param{gpi} has no gradient"
                )
            fx_inputs.append(param.grad)
        elif role == "state":
            param = optimizer.param_groups[gi]["params"][gpi]
            state_key = info["state_key"]
            fx_inputs.append(optimizer.state[param][state_key])
        elif role == "optimizer_attr":
            fx_inputs.append(getattr(optimizer, info["attr_name"]))
        else:
            # Unknown slot — use frozen value if available
            frozen = info.get("_frozen_value")
            if frozen is not None:
                fx_inputs.append(frozen)
            else:
                raise RuntimeError(
                    f"Optimizer replay: unknown slot {info.get('slot_index')} "
                    f"with no frozen value"
                )

    # Run the captured forward and write mutations back
    with torch.no_grad():
        result = replay.forward_fn(*fx_inputs)

        # Write mutations back via copy_()
        if replay.mutated_slot_indices:
            outputs = result if isinstance(result, (tuple, list)) else (result,)
            for out_idx, slot_idx in enumerate(replay.mutated_slot_indices):
                if out_idx < len(outputs) and isinstance(outputs[out_idx], torch.Tensor):
                    fx_inputs[slot_idx].copy_(outputs[out_idx])


def _match_arg_to_optimizer(arg, optimizer) -> dict:
    """Match a function call argument to its source in the optimizer.

    Used during inner fn recording to identify what each argument represents:
    param, grad, state tensor, optimizer attr (0-D scalar), stacked tensor,
    or a baked-in constant (int/float).
    """
    if not isinstance(arg, torch.Tensor):
        return {"role": "constant", "value": arg}

    try:
        ptr = arg.data_ptr()
    except Exception:
        return {"role": "unknown"}

    # Check params, grads, and state
    for gi, group in enumerate(optimizer.param_groups):
        for gpi, param in enumerate(group["params"]):
            try:
                if param.data_ptr() == ptr:
                    return {"role": "param", "group_index": gi, "group_param_index": gpi}
            except Exception:
                pass
            if param.grad is not None:
                try:
                    if param.grad.data_ptr() == ptr:
                        return {"role": "grad", "group_index": gi, "group_param_index": gpi}
                except Exception:
                    pass
            state = optimizer.state.get(param, {})
            for state_key, val in state.items():
                if isinstance(val, torch.Tensor):
                    try:
                        if val.data_ptr() == ptr:
                            return {
                                "role": "state",
                                "group_index": gi,
                                "group_param_index": gpi,
                                "state_key": str(state_key),
                            }
                    except Exception:
                        pass

    # Check ALL optimizer instance attributes (including _ prefixed ones
    # like _adamw_step_t, _muon_lr_t etc.)
    for attr_name in dir(optimizer):
        if attr_name.startswith("__"):
            continue
        try:
            val = getattr(optimizer, attr_name)
        except Exception:
            continue
        if isinstance(val, torch.Tensor):
            try:
                if val.data_ptr() == ptr:
                    return {"role": "optimizer_attr", "attr_name": attr_name}
            except Exception:
                continue

    # Check for stacked tensors (e.g. torch.stack(params) for muon).
    # These are freshly-created tensors whose data_ptr won't match anything.
    # Detect by shape: (num_params, *param_shape) for some param group.
    if arg.dim() > 0:
        for gi, group in enumerate(optimizer.param_groups):
            params = group["params"]
            if not params:
                continue
            first = params[0]
            if (arg.dim() == first.dim() + 1
                    and arg.shape[0] == len(params)
                    and arg.shape[1:] == first.shape):
                # Verify by comparing first slice value (before mutations)
                try:
                    if torch.equal(arg[0], first):
                        return {"role": "stacked_params", "group_index": gi}
                except Exception:
                    pass
                try:
                    if first.grad is not None and torch.equal(arg[0], first.grad):
                        return {"role": "stacked_grads", "group_index": gi}
                except Exception:
                    pass

    return {"role": "unknown"}


def _build_inner_replay_plan(optimizer, records: list[dict]) -> _InnerFnReplayPlan:
    """Build a replay plan from recorded inner fn calls.

    Each record has a proxy reference and pre-matched arg_roles.  We detect
    step-related optimizer attrs (by name) and stacked params that need
    copy-back after the inner fn modifies them.
    """
    calls = []
    step_attr_names = set()

    for record in records:
        arg_roles = record["arg_roles"]
        copy_back_groups = []

        for role in arg_roles:
            if role["role"] == "optimizer_attr":
                attr_name = role["attr_name"]
                if "step" in attr_name.lower():
                    step_attr_names.add(attr_name)
            if role["role"] == "stacked_params":
                copy_back_groups.append(role["group_index"])

        calls.append({
            "proxy": record["proxy"],
            "arg_roles": arg_roles,
            "copy_back_groups": copy_back_groups,
        })

    return _InnerFnReplayPlan(
        optimizer=optimizer,
        calls=calls,
        step_attr_names=list(step_attr_names),
    )


def _run_inner_replay(plan: _InnerFnReplayPlan) -> None:
    """Execute one optimizer step using the inner fn replay plan.

    Replaces the original optimizer.step() entirely.  Manages step counters,
    fills scalar tensors, stacks/unstacks params, and calls each inner fn's
    captured aten graph through its proxy.
    """
    optimizer = plan.optimizer

    # 1. Increment step counters for all params that have them
    new_step = None
    for group in optimizer.param_groups:
        for param in group["params"]:
            state = optimizer.state.get(param, {})
            if "step" in state:
                state["step"] += 1
                if new_step is None:
                    new_step = state["step"]

    # Build set for efficient lookup in per-call loop
    step_attr_set = set(plan.step_attr_names)

    # 3. Execute each inner fn call
    with torch.no_grad():
        for call in plan.calls:
            proxy = call["proxy"]
            arg_roles = call["arg_roles"]

            # Restore per-call optimizer attr values (e.g. muon LR prescaling
            # fills _muon_lr_t differently for each param group).
            for role in arg_roles:
                if role["role"] == "optimizer_attr" and "captured_value" in role:
                    attr = getattr(optimizer, role["attr_name"])
                    val = role["captured_value"]
                    if isinstance(val, (int, float)):
                        attr.fill_(val)
                    else:
                        attr.copy_(val)

            # Re-apply step attr overrides (they were restored to step-1
            # values above, now set to current step)
            if new_step is not None:
                for role in arg_roles:
                    if (role["role"] == "optimizer_attr"
                            and role.get("attr_name", "") in step_attr_set):
                        getattr(optimizer, role["attr_name"]).fill_(new_step)

            # Assemble args from live state
            args = []
            for role in arg_roles:
                r = role["role"]
                if r == "param":
                    gi, gpi = role["group_index"], role["group_param_index"]
                    args.append(optimizer.param_groups[gi]["params"][gpi])
                elif r == "grad":
                    gi, gpi = role["group_index"], role["group_param_index"]
                    param = optimizer.param_groups[gi]["params"][gpi]
                    if param.grad is None:
                        raise RuntimeError(
                            f"Inner fn replay: no gradient for "
                            f"group{gi}.param{gpi}"
                        )
                    args.append(param.grad)
                elif r == "state":
                    gi, gpi = role["group_index"], role["group_param_index"]
                    param = optimizer.param_groups[gi]["params"][gpi]
                    args.append(optimizer.state[param][role["state_key"]])
                elif r == "optimizer_attr":
                    args.append(getattr(optimizer, role["attr_name"]))
                elif r == "stacked_params":
                    gi = role["group_index"]
                    params = optimizer.param_groups[gi]["params"]
                    args.append(torch.stack(list(params)))
                elif r == "stacked_grads":
                    gi = role["group_index"]
                    params = optimizer.param_groups[gi]["params"]
                    args.append(torch.stack([p.grad for p in params]))
                elif r == "constant":
                    args.append(role["value"])
                else:
                    raise RuntimeError(f"Inner fn replay: unknown role '{r}'")

            # Call the proxy (replays captured aten, handles mutations)
            proxy(*tuple(args))

            # Copy back stacked params to individual params.
            # The proxy wrote mutation outputs back to the stacked tensor
            # via copy_(), but those are views of the NEW stacked tensor,
            # not the original individual params.
            for gi in call["copy_back_groups"]:
                params = list(optimizer.param_groups[gi]["params"])
                for i, role in enumerate(arg_roles):
                    if (role["role"] == "stacked_params"
                            and role["group_index"] == gi):
                        torch._foreach_copy_(
                            params, list(args[i].unbind(0))
                        )
                        break


# -----------------------------------------------------------------------------
# Optimizer auto-capture
# -----------------------------------------------------------------------------

_real_optimizer_init: Callable | None = None
_optimizer_captured: dict[int, dict] = {}  # id(optimizer) → capture info
_registered_optimizers: dict[int, dict] = {}  # id(obj) → {"obj": obj, "step_fn": fn}
_patched_optimizer_instances: dict[int, object] = {}  # id(optimizer) → optimizer (for unpatch)


def _stable_optimizer_value(value):
    """Serialize optimizer config/state values into a stable cache-key fragment."""
    if isinstance(value, torch.Tensor):
        return ("tensor", tuple(value.shape), str(value.dtype), str(value.device))
    if isinstance(value, (list, tuple)):
        return tuple(_stable_optimizer_value(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((str(k), _stable_optimizer_value(v)) for k, v in value.items()))
    if isinstance(value, (int, float, bool, str, type(None))):
        return value
    return (type(value).__qualname__, repr(value))


def _optimizer_cache_key(optimizer, step_fn: Callable | None = None) -> str:
    """Build a cache key for optimizer captures from layout + hyperparameters."""
    parts = [type(optimizer).__qualname__]

    if hasattr(optimizer, "param_groups"):
        for group in optimizer.param_groups:
            group_cfg = tuple(sorted(
                (key, _stable_optimizer_value(value))
                for key, value in group.items()
                if key != "params"
            ))
            parts.append(f"group:{group_cfg!r}")
            for param in group["params"]:
                parts.append(
                    "param:"
                    f"{tuple(param.shape)}:{param.dtype}:{param.device}:{bool(param.requires_grad)}"
                )
                if hasattr(optimizer, "state"):
                    state = optimizer.state.get(param, {})
                    if state:
                        parts.append(f"state:{_stable_optimizer_value(state)!r}")
    else:
        target = step_fn or getattr(optimizer, "step", None)
        if target is not None:
            try:
                src_lines, _ = inspect.getsourcelines(target)
                step_src = "".join(src_lines)
            except (OSError, TypeError):
                step_src = repr(target)
            parts.append(f"step:{step_src}")

    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]
    return f"optimizer_{type(optimizer).__name__}_{digest}"


def register_optimizer(
    optimizer,
    step_fn: Callable | None = None,
    param_name_map: dict[int, str] | None = None,
) -> None:
    """Register a non-standard optimizer for auto-capture.

    For optimizers that don't inherit from ``torch.optim.Optimizer``
    (e.g. NorMuonAndAdam), this is a lightweight escape hatch — one line
    instead of a full recipe.

    Example::

        _orig_step = optimizer.step  # capture before register_optimizer patches it
        torch_graph.auto_install.register_optimizer(
            optimizer,
            step_fn=lambda: _orig_step(do_adam=True),
            param_name_map={id(p): name for name, p in model.named_parameters()},
        )

    After registration, the optimizer's step will be auto-captured on
    first call (just like standard optimizers).  The original ``step``
    method is monkey-patched so the capture happens transparently.

    Args:
        optimizer: The optimizer object.
        step_fn: Zero-arg callable that performs one step.  If None,
            ``optimizer.step`` is used.
        param_name_map: Optional ``id(param) → name`` mapping. When omitted,
            optimizer slots use stable names like ``group0.param3`` instead
            of guessing model-owned parameter names.
    """
    actual_step = step_fn or optimizer.step
    opt_id = id(optimizer)

    _registered_optimizers[opt_id] = {
        "obj": optimizer,
        "step_fn": actual_step,
        "original_step": optimizer.step,
        "param_name_map": param_name_map,
    }

    # Monkey-patch this specific instance's step method
    def _wrapped_step(*args, **kwargs):
        def _run_registered_step():
            if step_fn is not None and not args and not kwargs:
                return actual_step()
            return _registered_optimizers[opt_id]["original_step"](*args, **kwargs)

        if not _config.capture_optimizer:
            return _run_registered_step()

        if opt_id in _optimizer_captured:
            return _run_registered_step()

        # First call: capture using the registered step_fn (step runs inside capture)
        return _do_optimizer_capture(
            optimizer,
            step_fn=_run_registered_step,
            param_name_map=param_name_map,
        )

    # Stash original so _do_optimizer_capture's restore logic can temporarily
    # un-patch the step during capture, preventing lambda→wrapper recursion.
    optimizer._torch_graph_original_step = optimizer.step
    optimizer.step = _wrapped_step
    _registered_optimizers[opt_id]["wrapped_step"] = _wrapped_step

    if _config.verbose:
        logger.info(
            f"Registered optimizer {type(optimizer).__name__} for auto-capture"
        )


def _has_inner_compiled_fns(optimizer) -> bool:
    """Check if an optimizer's step uses inner @torch.compile-decorated functions.

    Scans the optimizer's module for _CompiledFnProxy instances, which are
    created when our patched torch.compile wraps a standalone function.
    This detects optimizers like MuonAdamW that use @torch.compile on
    inner helper functions (adamw_step_fused, muon_step_fused).
    """
    # Get the module where the optimizer's step method is defined
    step_method = getattr(optimizer, '_torch_graph_original_step', None)
    if step_method is None:
        step_method = getattr(type(optimizer), 'step', None)
    if step_method is None:
        return False

    # Check the module's globals for _CompiledFnProxy instances
    try:
        mod = inspect.getmodule(step_method)
    except Exception:
        return False
    if mod is None:
        return False

    for attr_name in dir(mod):
        try:
            val = getattr(mod, attr_name)
        except Exception:
            continue
        if isinstance(val, _CompiledFnProxy):
            return True

    return False


def _wrap_optimizer_step(optimizer):
    """Wrap an optimizer's step method for auto-capture on first call.

    Every optimizer subclass (SGD, Adam, etc.) overrides Optimizer.step,
    so we can't patch the base class.  Instead, we wrap the instance's
    bound step method after __init__ completes.
    """
    original_step = optimizer.step
    opt_id = id(optimizer)

    def _auto_capture_step(*args, **kwargs):
        global _capture_depth, _recording_inner_calls, _recording_optimizer
        if not _config.capture_optimizer or opt_id in _optimizer_captured:
            # Check if we have a replay function for this optimizer
            entry = _optimizer_captured.get(opt_id)
            replay = entry.get("replay") if entry else None
            if replay is not None:
                _capture_depth += 1
                try:
                    _run_optimizer_replay(replay)
                    return None
                finally:
                    _capture_depth -= 1

            # If this optimizer uses inner compiled functions (e.g. MuonAdamW),
            # check for inner replay plan (full replay without original code).
            if _optimizer_captured.get(opt_id, {}).get("uses_inner_compiled"):
                inner_replay = _optimizer_captured[opt_id].get("inner_replay")
                if inner_replay is not None:
                    _run_inner_replay(inner_replay)
                    return None
                # No replay plan — run eagerly, inner proxies replay on their own
                return original_step(*args, **kwargs)

            # Run eagerly but suppress _CompiledFnProxy capture for any
            # @torch.compile-decorated helpers inside the optimizer.
            _capture_depth += 1
            try:
                return original_step(*args, **kwargs)
            finally:
                _capture_depth -= 1

        # Check if the optimizer uses inner @torch.compile-decorated functions.
        # If so, skip monolithic capture — let the inner proxies capture/replay
        # individually while the outer Python step runs eagerly.
        if _has_inner_compiled_fns(optimizer):
            if _config.verbose:
                logger.info(
                    f"Optimizer {type(optimizer).__name__} uses inner "
                    f"@torch.compile functions. Skipping monolithic capture."
                )

            # Record inner fn calls for building replay plan.
            # Matching happens during recording (before mutations).
            _recording_inner_calls = True
            _recording_optimizer = optimizer
            _inner_call_records.clear()
            try:
                result = original_step(*args, **kwargs)
            finally:
                _recording_inner_calls = False
                _recording_optimizer = None

            # Build inner fn replay plan if replay is enabled
            inner_replay = None
            if _config.replay_optimizer and _inner_call_records:
                inner_replay = _build_inner_replay_plan(
                    optimizer, _inner_call_records,
                )
                if _config.verbose:
                    n_calls = len(inner_replay.calls)
                    step_attrs = inner_replay.step_attr_names
                    logger.info(
                        f"Inner fn replay plan: {n_calls} calls, "
                        f"step attrs: {step_attrs}"
                    )
            _inner_call_records.clear()

            _optimizer_captured[opt_id] = {
                "source": "inner_compiled",
                "uses_inner_compiled": True,
                "inner_replay": inner_replay,
            }
            return result

        # No inner compiled fns — do monolithic capture
        return _do_optimizer_capture(optimizer, step_fn=None, step_args=args, step_kwargs=kwargs)

    optimizer.step = _auto_capture_step
    # Stash original so we can restore it
    optimizer._torch_graph_original_step = original_step
    _patched_optimizer_instances[opt_id] = optimizer


def _do_optimizer_capture(
    optimizer,
    step_fn: Callable | None = None,
    step_args: tuple = (),
    step_kwargs: dict | None = None,
    param_name_map: dict[int, str] | None = None,
):
    """Capture optimizer.step() aten graph and save to disk."""
    from torch_graph.export import capture_optimizer_aten, export_aten_program

    if step_kwargs is None:
        step_kwargs = {}

    opt_name = type(optimizer).__name__
    cache_key = _optimizer_cache_key(optimizer, step_fn=step_fn)
    cache_path = _cache_path(cache_key)
    opt_id = id(optimizer)

    # Check disk cache
    if cache_path.exists() and not _config.force_recapture:
        if _config.verbose:
            tag = " (user-modified)" if _has_user_modified(cache_path) else ""
            logger.info(f"Optimizer{tag} aten cached: {cache_path}")

        # Build replay from cached aten + meta if replay is enabled
        replay_info = None
        if _config.replay_optimizer:
            replay_info = _load_optimizer_replay(optimizer, cache_path)

        _optimizer_captured[opt_id] = {
            "cache_path": str(cache_path),
            "source": "cached",
            "replay": replay_info,
        }
        # Still need to run the step eagerly since we loaded from cache
        original_step = getattr(optimizer, '_torch_graph_original_step', optimizer.step)
        if step_fn is not None:
            return step_fn()
        return original_step(*step_args, **step_kwargs)

    if _config.verbose:
        logger.info(f"Capturing optimizer aten for {opt_name}...")

    # Default to reliable optimizer-slot names instead of guessing model names.
    resolved_param_name_map = dict(param_name_map or {})
    if hasattr(optimizer, 'param_groups'):
        for group_idx, group in enumerate(optimizer.param_groups):
            for group_param_idx, p in enumerate(group["params"]):
                resolved_param_name_map.setdefault(
                    id(p),
                    f"group{group_idx}.param{group_param_idx}",
                )

    # Restore original step during capture to avoid recursion
    original_step = getattr(optimizer, '_torch_graph_original_step', optimizer.step)
    if hasattr(optimizer, '_torch_graph_original_step'):
        optimizer.step = original_step
    # Build actual step callable, forwarding args/kwargs for e.g. LBFGS(closure)
    actual_step_fn = step_fn if step_fn is not None else (
        (lambda: original_step(*step_args, **step_kwargs))
        if (step_args or step_kwargs) else None
    )
    # When replay is enabled, we need real tensors to freeze unknown slots
    need_real_tensors = _config.record_real_tensors or _config.replay_optimizer
    _unpatch_for_capture()
    try:
        opt_capture = capture_optimizer_aten(
            optimizer,
            record_real_tensors=need_real_tensors,
            param_name_map=resolved_param_name_map,
            step_fn=actual_step_fn,
        )
    finally:
        _repatch_after_capture()
        # Re-wrap step (capture may have changed the method)
        if opt_id in _registered_optimizers:
            optimizer.step = _registered_optimizers[opt_id]["wrapped_step"]
        elif hasattr(optimizer, '_torch_graph_original_step'):
            _wrap_optimizer_step(optimizer)

    if not opt_capture.forward_graphs:
        logger.warning(f"Optimizer capture produced no graphs for {opt_name}")
        _optimizer_captured[opt_id] = {"source": "failed"}
        return getattr(opt_capture, "step_result", None)

    # Export to disk
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    export_aten_program(
        opt_capture, str(cache_path),
        include_test_harness=False,
        skip_pt=_config.skip_pt,
    )

    n_nodes = len(list(opt_capture.forward_graphs[0].graph_module.graph.nodes))
    if _config.verbose:
        logger.info(f"Saved optimizer aten ({n_nodes} nodes): {cache_path}")

    # Detect mutations and build slot_info for replay
    gm = opt_capture.forward_graphs[0].graph_module
    slot_info = (
        opt_capture.optimizer_slot_info[0]
        if opt_capture.optimizer_slot_info
        else []
    )

    # Enrich unknown slots using the now-populated optimizer state.
    # On first AdamW step, state (exp_avg, exp_avg_sq, step) is lazily
    # created INSIDE the compiled step, so the pre-capture catalog misses them.
    # After capture, optimizer.state is populated and we can re-match using
    # the live data_ptrs captured during the FX compilation pass.
    real_inputs = getattr(opt_capture, "forward_real_inputs", None)
    input_ptrs = getattr(opt_capture, "_optimizer_input_ptrs", None)
    if slot_info and hasattr(optimizer, "param_groups"):
        _enrich_unknown_slots(slot_info, real_inputs, optimizer, input_ptrs)

    # Count FX outputs (all are mutations for optimizer.step() returning None)
    out_node = next(n for n in gm.graph.nodes if n.op == "output")
    out_args = out_node.args[0]
    num_outputs = len(out_args) if isinstance(out_args, (tuple, list)) else 1

    mutated_slot_indices = _detect_mutated_slots(slot_info, num_outputs)

    # Serialize slot_info for .meta (strip non-serializable fields)
    serializable_slot_info = []
    for info in slot_info:
        clean = {k: v for k, v in info.items() if k != "_frozen_value"}
        serializable_slot_info.append(clean)

    _write_meta(cache_path, {
        "function": f"{opt_name}.step",
        "cache_key": cache_key,
        "optimizer_class": opt_name,
        "node_count": n_nodes,
        "mutated_slot_indices": mutated_slot_indices,
        "slot_info": serializable_slot_info,
    })

    _save_capture_artifacts(opt_capture, cache_path, "optimizer",
                            require_h5_functions=True)

    # Build replay info if enabled
    replay_info = None
    if _config.replay_optimizer and slot_info:
        # Freeze unknown slot values from capture for replay
        for i, info in enumerate(slot_info):
            if info.get("role") == "unknown":
                real_inputs = getattr(opt_capture, "forward_real_inputs", None)
                if real_inputs and i < len(real_inputs):
                    val = real_inputs[i]
                    if isinstance(val, torch.Tensor):
                        info["_frozen_value"] = val.clone().detach()
                    else:
                        info["_frozen_value"] = val
                    if _config.verbose:
                        logger.info(
                            f"Optimizer replay: freezing unknown slot {i} "
                            f"(shape={val.shape if isinstance(val, torch.Tensor) else 'scalar'})"
                        )

        replay_info = _build_optimizer_replay(
            optimizer, gm.forward, slot_info, mutated_slot_indices,
        )
        if _config.verbose:
            logger.info(
                f"Optimizer replay enabled: {len(slot_info)} slots, "
                f"{len(mutated_slot_indices)} mutations"
            )

    _optimizer_captured[opt_id] = {
        "cache_path": str(cache_path),
        "source": "captured",
        "replay": replay_info,
    }

    _installed[opt_id] = _InstalledEntry(
        name=f"{opt_name}.step",
        kind="optimizer",
        cache_path=str(cache_path),
        source="captured" if replay_info is None else "replay",
    )
    return getattr(opt_capture, "step_result", None)
def _patched_optimizer_init(self, *args, **kwargs):
    """Patched Optimizer.__init__: wraps step method for auto-capture."""
    _real_optimizer_init(self, *args, **kwargs)
    _wrap_optimizer_step(self)


def _patch_optimizer():
    """Monkey-patch Optimizer.__init__ to auto-wrap step methods."""
    global _real_optimizer_init
    if _real_optimizer_init is not None:
        return  # already patched
    _real_optimizer_init = torch.optim.Optimizer.__init__
    torch.optim.Optimizer.__init__ = _patched_optimizer_init


def _unpatch_optimizer():
    """Restore real Optimizer.__init__ and step on all already-patched instances."""
    global _real_optimizer_init
    if _real_optimizer_init is not None:
        torch.optim.Optimizer.__init__ = _real_optimizer_init
        _real_optimizer_init = None
    # Restore step method on all instances patched by _wrap_optimizer_step
    for opt in list(_patched_optimizer_instances.values()):
        if hasattr(opt, '_torch_graph_original_step'):
            opt.step = opt._torch_graph_original_step
            del opt._torch_graph_original_step
    _patched_optimizer_instances.clear()
    # Restore step on registered (non-standard) optimizer instances
    for info in _registered_optimizers.values():
        opt = info["obj"]
        if hasattr(opt, '_torch_graph_original_step'):
            opt.step = opt._torch_graph_original_step
            del opt._torch_graph_original_step
        elif "original_step" in info:
            opt.step = info["original_step"]
    _registered_optimizers.clear()


# -----------------------------------------------------------------------------
# The monkey-patch
# -----------------------------------------------------------------------------

def _patched_compile(model_or_fn=None, **kwargs):
    """Drop-in torch.compile replacement: proxy that captures+installs on first call."""
    # Handle the decorator case: @torch.compile or @torch.compile(...)
    if model_or_fn is None:
        # Called as @torch.compile(...) → return a decorator
        def decorator(fn):
            return _patched_compile(fn, **kwargs)
        return decorator

    if isinstance(model_or_fn, nn.Module):
        return _CompiledModelProxy(model_or_fn, kwargs)
    elif callable(model_or_fn):
        return _CompiledFnProxy(model_or_fn, kwargs)
    else:
        raise TypeError(
            f"torch.compile expects nn.Module or callable, got {type(model_or_fn)}"
        )


def patch() -> None:
    """Monkey-patch torch.compile and Optimizer.step. Called automatically on import."""
    global _real_torch_compile
    if _real_torch_compile is not None:
        return  # already patched

    _real_torch_compile = torch.compile
    torch.compile = _patched_compile
    _patch_optimizer()

    if _config.verbose:
        logger.info("Patched torch.compile + Optimizer.step → aten capture/install")


def unpatch() -> None:
    """Restore the original torch.compile and Optimizer.__init__."""
    global _real_torch_compile, _capture_depth, _call_count
    global _recording_inner_calls, _recording_optimizer
    _capture_depth = 0
    _call_count = 0
    _recording_inner_calls = False
    _recording_optimizer = None
    _inner_call_records.clear()
    _step_losses.clear()
    if _real_torch_compile is not None:
        torch.compile = _real_torch_compile
        _real_torch_compile = None
    _unpatch_optimizer()
    if _config.verbose:
        logger.info("Restored original torch.compile + Optimizer.__init__")


def status() -> str:
    """Return a summary of all installed aten replacements."""
    if not _installed:
        return "No aten replacements installed."
    lines = [f"{len(_installed)} aten replacement(s) installed:"]
    for entry in _installed.values():
        src = entry.source
        if entry.cache_path:
            src += f" ({entry.cache_path})"
        lines.append(f"  [{entry.kind}] {entry.name}: {src}")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Explicit install API (for when you don't want monkey-patching)
# -----------------------------------------------------------------------------

def install_from_file(
    model: nn.Module,
    aten_path: str | Path,
    *,
    num_real_outputs: int = 1,
    validate: bool = True,
) -> None:
    """Load a specific aten .py file and install it onto a model (no caching)."""
    path = Path(aten_path)
    if not path.exists():
        raise FileNotFoundError(f"Aten file not found: {path}")

    aten_mod = _load_aten_module(path)

    from torch_graph.install import _parse_param_paths
    param_paths = _parse_param_paths(aten_mod)

    if not param_paths:
        raise ValueError(
            f"Cannot determine parameter mapping from {path}. "
            f"Ensure the file has a 'Parameter mapping:' section."
        )

    if validate:
        _validate_model_shapes(model, aten_mod, param_paths)

    num_real = num_real_outputs
    if hasattr(aten_mod, 'NUM_REAL_OUTPUTS'):
        num_real = aten_mod.NUM_REAL_OUTPUTS

    _install_model_from_module(
        model, aten_mod,
        param_paths=param_paths,
        num_real_outputs=num_real,
    )

    _installed[id(model)] = _InstalledEntry(
        name=type(model).__name__,
        kind="model",
        cache_path=str(path),
        source="loaded_from_disk",
        param_paths=param_paths,
    )

    if _config.verbose:
        logger.info(f"Installed {path} onto {type(model).__name__}")


def install_fn_from_file(
    aten_path: str | Path,
    *,
    fn_name: str | None = None,
) -> Callable:
    """Load a function from an aten .py file. Tries fn_name, then forward(), then first public callable.

    Returns the loaded callable — caller is responsible for monkey-patching
    it into the right place.
    """
    path = Path(aten_path)
    if not path.exists():
        raise FileNotFoundError(f"Aten file not found: {path}")

    aten_mod = _load_aten_module(path)

    if fn_name and hasattr(aten_mod, fn_name):
        return getattr(aten_mod, fn_name)
    if hasattr(aten_mod, 'forward'):
        return aten_mod.forward

    for attr_name in dir(aten_mod):
        if attr_name.startswith('_'):
            continue
        attr = getattr(aten_mod, attr_name)
        if callable(attr):
            return attr

    raise ValueError(f"No callable found in {path}")


# -----------------------------------------------------------------------------
# Auto-patch on import
# -----------------------------------------------------------------------------

patch()
