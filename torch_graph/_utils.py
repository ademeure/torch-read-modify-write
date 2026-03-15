"""Shared utilities for FakeTensor handling, tensor materialization, and FX interpretation."""

from __future__ import annotations

import re

import torch
from torch.fx import GraphModule
from torch.fx.interpreter import Interpreter


# Pre-compiled regexes for FX name shortening
_PRIMALS_RE = re.compile(r"^primals_(\d+)$")
_TANGENTS_RE = re.compile(r"^tangents_(\d+)$")
# Pre-compiled regexes for module path cleaning (clean_self_path)
_MODULES_DICT_RE = re.compile(r"_modules\['([^']*)'\]")
_L_BRACKET_RE = re.compile(r"L\['([^']*)'\]")

try:
    from torch._subclasses.fake_tensor import is_fake as _torch_is_fake
except ImportError:
    _torch_is_fake = None


def is_fake(t: torch.Tensor) -> bool:
    """Check if a tensor is a FakeTensor (no real data, used during tracing).

    IMPORTANT: In PyTorch 2.10+, FakeTensors look more like real tensors —
    untyped_storage() succeeds and tolist() returns symbolic names (zuf0, ...).
    Always use this function before calling .data_ptr(), .tolist(), .numpy(),
    or pickle on a tensor that might be fake.

    Falls back through multiple heuristics for compatibility across PyTorch versions.
    """
    if _torch_is_fake is not None:
        try:
            return _torch_is_fake(t)
        except Exception:
            pass
    if type(t).__name__ == "FakeTensor":
        return True
    if hasattr(t, "device") and t.device.type == "meta":
        return True
    if not hasattr(t, "storage"):
        return True
    if hasattr(t, "_has_symbolic_sizes_strides") and t._has_symbolic_sizes_strides:
        return True
    return False


def _target_device(val) -> torch.device:
    """Infer the target device from a tensor or FakeTensor meta val.

    Returns CPU for meta-device or non-tensor inputs.
    """
    if hasattr(val, "device"):
        d = val.device
        if d.type != "meta":
            return d
    return torch.device("cpu")


def materialize_tensor(t: torch.Tensor) -> torch.Tensor:
    """Convert any tensor (including FakeTensor/subclasses) to a real, picklable tensor.

    Tries progressively more aggressive strategies:
      1. clone (works for real tensors)
      2. empty + copy_ (works for some FakeTensor subclasses)
      3. randn/zeros with matching shape (last resort — loses data but preserves shape)

    Preserves the original device (e.g. CUDA tensors stay on CUDA).
    """
    shape, dtype, device = list(t.shape), t.dtype, _target_device(t)

    if not is_fake(t):
        try:
            return t.detach().clone()
        except Exception:
            pass  # unusual subclass — try copy_ approach

    try:
        fresh = torch.empty(shape, dtype=dtype, device=device)
        with torch.no_grad():
            fresh.copy_(t.detach())
        return fresh
    except Exception:
        pass  # FakeTensor — can't copy data, fall through to random init

    return (torch.randn(shape, dtype=dtype, device=device) if dtype.is_floating_point
            else torch.zeros(shape, dtype=dtype, device=device))


def clean_self_path(path: str, keep_self: bool = True) -> str:
    """Clean FX-generated module paths by removing or simplifying L['self'].

    Also normalizes ``_modules['X']`` dict-access syntax (produced by
    aot_autograd in PyTorch <2.10) into plain dotted attribute names.

    Args:
        path: FX module path like "L['self'].transformer.h.0"
              or "L['self']._modules['layers']._modules['0']"
        keep_self: If True, replace with "self." prefix. If False, strip entirely.
    """
    # Normalize _modules['X'] → X  (PyTorch <2.10 aot_autograd format)
    path = _MODULES_DICT_RE.sub(r"\1", path)
    # Clean up any resulting double dots from "._modules['X']" → ".X"
    path = path.replace("..", ".")
    if keep_self:
        # L['self'] → self, L['fn'] / L['other'] → fn / other
        path = path.replace("L['self'].", "self.").replace("L['self']", "self")
        return _L_BRACKET_RE.sub(r"\1", path)
    else:
        path = path.replace("L['self'].", "").replace("L['self']", "")
        return _L_BRACKET_RE.sub(r"\1", path)


def short_name(name: str) -> str:
    """Shorten FX names: primals_5 -> p5, tangents_3 -> d3."""
    if m := _PRIMALS_RE.match(name): return f"p{m.group(1)}"
    if m := _TANGENTS_RE.match(name): return f"d{m.group(1)}"
    return name


def h5_load_function_source(map_short_to_long: bool = False) -> str:
    """Return inline Python source for a ``_load_h5(path, keys=None)`` function.

    The generated code expects ``torch`` and ``_device`` to already be defined.
    It handles bfloat16, float16, float8 (raw_uint8) and 0-dim scalars.

    Args:
        map_short_to_long: Also create ``primals_N``/``tangents_N`` aliases
            for short H5 names (``p6`` → ``primals_6``, ``d3`` → ``tangents_3``).
    """
    lines = [
        'def _load_h5(path, keys=None):',
        '    import h5py',
        '    _DTYPE = {"torch.bfloat16": torch.bfloat16, "torch.float16": torch.float16}',
        '    w = {}',
        '    with h5py.File(path, "r") as f:',
        '        grp = f["tensors"]',
        '        for name in (keys if keys is not None else grp):',
        '            ds = grp[name]',
        '            raw = ds[()]',
        '            val = torch.tensor(raw).to(_device) if raw.ndim == 0 else torch.from_numpy(raw).to(_device)',
        '            orig = ds.attrs.get("torch_dtype", "")',
        '            stored_as = ds.attrs.get("stored_as", "")',
        '            if "raw_uint8" in stored_as and orig:',
        '                val = val.view(getattr(torch, orig.replace("torch.", ""), val.dtype))',
        '            elif orig in _DTYPE:',
        '                val = val.to(_DTYPE[orig])',
        '            w[name] = val',
    ]
    if map_short_to_long:
        lines += [
            '            import re as _re',
            '            _m = _re.match(r"^p(\\d+)$", name)',
            '            if _m: w[f"primals_{_m.group(1)}"] = val',
            '            _m = _re.match(r"^d(\\d+)$", name)',
            '            if _m: w[f"tangents_{_m.group(1)}"] = val',
        ]
    lines.append('    return w')
    return '\n'.join(lines) + '\n'


def load_h5_tensors(
    path: str,
    keys: list[str] | None = None,
    device: torch.device | str | None = None,
) -> dict[str, torch.Tensor]:
    """Load tensors from an H5 file, restoring original dtypes.

    Handles bfloat16 (stored as float32), float16 (stored as float32),
    float8 variants (stored as raw_uint8), and 0-dim scalars.

    Args:
        path: Path to the H5 file.
        keys: Optional list of tensor names to load.  When ``None``, all
            tensors in the ``/tensors`` group are loaded.
        device: Optional device to move tensors to.  ``None`` keeps them
            on CPU (the default for ``torch.from_numpy``).

    Returns:
        Dictionary mapping tensor names to restored ``torch.Tensor`` values.
    """
    import h5py
    import numpy as np

    _DTYPE_MAP = {
        "torch.bfloat16": torch.bfloat16,
        "torch.float16": torch.float16,
        "torch.float8_e4m3fn": torch.float8_e4m3fn,
        "torch.float8_e5m2": torch.float8_e5m2,
    }

    result: dict[str, torch.Tensor] = {}
    with h5py.File(path, "r") as f:
        if "tensors" not in f:
            return result
        grp = f["tensors"]
        names = keys if keys is not None else list(grp)
        for name in names:
            ds = grp[name]
            raw = ds[()]
            # 0-dim scalars: np.generic, not ndarray — torch.from_numpy rejects them
            if isinstance(raw, np.ndarray):
                val = torch.from_numpy(raw)
            else:
                val = torch.tensor(raw)

            orig_dtype = ds.attrs.get("torch_dtype", "")
            stored_as = ds.attrs.get("stored_as", "")

            if orig_dtype in _DTYPE_MAP:
                target_dt = _DTYPE_MAP[orig_dtype]
                if "raw_uint8" in stored_as:
                    val = val.view(target_dt)
                elif orig_dtype == "torch.bfloat16" and val.dtype != torch.bfloat16:
                    # float32 -> uint16 bit-reinterpret -> bfloat16 is lossless
                    # but simple .to(bfloat16) also works since bottom 16 bits are zero
                    val = val.to(torch.bfloat16)
                else:
                    val = val.to(target_dt)

            if device is not None:
                val = val.to(device)

            result[name] = val
    return result


class RecordingInterpreter(Interpreter):
    """FX Interpreter that records every intermediate tensor value.

    Args:
        gm: The GraphModule to interpret.
        record_nodes: Optional set of node names to record.  When provided,
            only nodes whose name is in this set will have their tensors
            cloned and stored.  All nodes are still *executed* for
            correctness; only the recording is skipped for non-matching
            nodes.  When ``None`` (default), all nodes are recorded.
        skip_views: If True (default), skip recording view/getitem/reshape
            and other zero-copy ops whose outputs share storage with inputs.
    """

    # Ops that are reshapes, views, or low-value plumbing — not worth recording
    _VIEW_OPS = frozenset({
        "view", "t", "transpose", "reshape", "permute", "expand",
        "contiguous", "unsqueeze", "squeeze", "getitem",
        "clone", "detach", "split", "slice", "select", "as_strided",
    })

    def __init__(
        self,
        gm: GraphModule,
        record_nodes: set[str] | None = None,
        skip_views: bool = False,
    ):
        super().__init__(gm)
        self.recorded: dict[str, torch.Tensor] = {}
        self.final_output = None
        self._record_nodes = record_nodes
        self._skip_views = skip_views

    @staticmethod
    def _is_view_op(n) -> bool:
        """Check if a node is a view/getitem/reshape op."""
        target = str(n.target)
        # Extract short op name from e.g. "aten.view.default" or "operator.getitem"
        short = target.rsplit(".", 1)[-1] if "." in target else target
        # Also check the part before .default/.out etc.
        parts = target.split(".")
        for p in parts:
            if p in RecordingInterpreter._VIEW_OPS:
                return True
        return short in RecordingInterpreter._VIEW_OPS

    def run_node(self, n):
        result = super().run_node(n)
        if n.op == "output":
            self.final_output = result
        if self._record_nodes is not None and n.name not in self._record_nodes:
            return result
        if self._skip_views and n.op == "call_function" and self._is_view_op(n):
            return result
        if isinstance(result, torch.Tensor):
            try:
                self.recorded[n.name] = result.clone().detach()
            except Exception:
                self.recorded[n.name] = materialize_tensor(result)
        elif isinstance(result, (tuple, list)):
            # Store first tensor as the representative output. Don't use "{name}_{i}" —
            # that collides with FX's own node naming (native_layer_norm_1, _2 etc.).
            # Individual tuple elements are recorded via downstream getitem nodes.
            for elem in result:
                if isinstance(elem, torch.Tensor):
                    try:
                        self.recorded[n.name] = elem.clone().detach()
                    except Exception:
                        self.recorded[n.name] = materialize_tensor(elem)
                    break
        return result
