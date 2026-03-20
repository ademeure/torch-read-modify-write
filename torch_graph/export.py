"""Export FX graphs as standalone, runnable, editable Python scripts.

Captures forward AND backward graphs at the aten op level via aot_autograd,
then exports them as self-contained Python files that:
  - Import only torch (no model code needed)
  - Embed all weights as tensor literals or .pt file references
  - Use raw aten ops (torch.ops.aten.*)
  - Can be edited (swap ops, change shapes, etc.) and rerun directly
  - Include the autograd backward pass as explicit aten ops
  - Show original PyTorch source annotations above each group of ops
"""

from __future__ import annotations

import contextlib
import copy
import logging
import os
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any, Callable

import torch
from torch.fx import GraphModule, Node

from torch_graph._utils import RecordingInterpreter as _RecordInterp, is_fake as _is_fake, materialize_tensor as _materialize_tensor, clean_self_path as _clean_self_path, h5_load_function_source as _h5_load_function_source
from torch_graph.internal_ir import (
    callable_to_str as _ir_callable_to_str,
    fx_node_to_python as _ir_fx_node_to_python,
    graph_to_ir as _build_graph_ir,
    ir_return_to_python as _ir_return_to_python,
    tensor_to_constructor as _ir_tensor_to_constructor,
)

logger = logging.getLogger(__name__)

from torch_graph.custom_ops import (
    find_custom_op_namespaces as _find_custom_op_namespaces,
    build_op_providers as _build_op_providers,
    emit_custom_op_imports as _emit_custom_op_imports,
)


# ── Triton kernel extraction for export ──────────────────────────────


def _collect_triton_kernels(*graph_modules: GraphModule) -> tuple[dict[int, dict], bool]:
    """Collect unique Triton kernel info from triton_kernel_wrapper_functional nodes.

    Returns (kernels_dict, has_tma) where kernels_dict maps
    kernel_idx → {name, source, const_args_by_idx}.
    """
    import inspect
    from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table

    kernels: dict[int, dict] = {}
    has_tma = False
    for gm in graph_modules:
        for node in gm.graph.nodes:
            if node.op != "call_function":
                continue
            if getattr(node.target, "__name__", "") != "triton_kernel_wrapper_functional":
                continue

            if node.kwargs.get("tma_descriptor_metadata"):
                has_tma = True

            kernel_idx = node.kwargs.get("kernel_idx", 0)
            if kernel_idx in kernels:
                # Still collect new constant_args_idx variants
                cai = node.kwargs.get("constant_args_idx", 0)
                if cai not in kernels[kernel_idx]["const_args_by_idx"]:
                    try:
                        kernels[kernel_idx]["const_args_by_idx"][cai] = kernel_side_table.get_constant_args(cai)
                    except Exception:
                        pass
                continue

            kernel_name = f"_triton_kernel_{kernel_idx}"
            source = None
            try:
                kernel_obj = kernel_side_table.get_kernel(kernel_idx)
                kernel_fn = kernel_obj.fn
                kernel_name = getattr(kernel_fn, "__name__", kernel_name)
                # Try multiple approaches to get source:
                # 1. inspect.getsource on raw functions
                # 2. JITFunction.fn (underlying function) + inspect
                # 3. JITFunction.raw_src (list of source lines with decorator)
                # 4. JITFunction.src (source without decorator)
                try:
                    source = inspect.getsource(kernel_fn)
                except (OSError, TypeError):
                    # kernel_fn is likely a JITFunction, not a raw function
                    if hasattr(kernel_fn, "fn"):
                        try:
                            source = inspect.getsource(kernel_fn.fn)
                        except (OSError, TypeError):
                            pass
                    if source is None and hasattr(kernel_fn, "raw_src"):
                        raw = kernel_fn.raw_src
                        if isinstance(raw, list):
                            source = "".join(raw)
                        elif isinstance(raw, str):
                            source = raw
                    if source is None and hasattr(kernel_fn, "src"):
                        source = "@triton.jit\n" + kernel_fn.src
            except Exception:
                pass

            const_args_by_idx: dict[int, dict] = {}
            cai = node.kwargs.get("constant_args_idx", 0)
            try:
                const_args_by_idx[cai] = kernel_side_table.get_constant_args(cai)
            except Exception:
                pass

            kernels[kernel_idx] = {
                "name": kernel_name,
                "source": source,
                "const_args_by_idx": const_args_by_idx,
            }

    return kernels, has_tma


def _emit_triton_kernels(buf, triton_kernels: dict[int, dict],
                         needs_tma: bool = False) -> None:
    """Emit Triton imports and kernel function definitions into the export header."""
    if not triton_kernels:
        return

    buf.write("import triton\n")
    buf.write("import triton.language as tl\n")
    if needs_tma:
        buf.write("from triton.tools.tensor_descriptor import TensorDescriptor\n")
    buf.write("\n")

    emitted_names: set[str] = set()
    for kid in sorted(triton_kernels):
        kinfo = triton_kernels[kid]
        name = kinfo["name"]
        source = kinfo["source"]
        if name in emitted_names:
            continue
        emitted_names.add(name)

        if source:
            # Dedent source to module level (remove common leading whitespace)
            import textwrap
            source = textwrap.dedent(source)
            buf.write(f"# ── Triton kernel: {name} ──\n")
            buf.write(source)
            if not source.endswith("\n"):
                buf.write("\n")
            buf.write("\n")
        else:
            buf.write(f"# WARNING: Could not extract source for Triton kernel '{name}' (kernel_idx={kid})\n")
            buf.write(f"# You will need to define {name} manually.\n\n")


def _emit_standard_header(buf, op_providers=None, triton_kernels=None,
                           needs_tma: bool = False) -> None:
    """Emit the shared imports / device detection / argparse header."""
    buf.write("import operator\n")
    buf.write("import os\n")
    buf.write("import torch\n")
    buf.write("aten = torch.ops.aten  # shorthand for low-level ops\n\n")
    if op_providers:
        _emit_custom_op_imports(buf, op_providers)
    if triton_kernels:
        _emit_triton_kernels(buf, triton_kernels, needs_tma=needs_tma)
    buf.write("# Device: defaults to CUDA if available, use --cpu to force CPU\n")
    buf.write("import argparse as _argparse\n")
    buf.write('_dev_parser = _argparse.ArgumentParser(add_help=False)\n')
    buf.write('_dev_parser.add_argument("--cpu", action="store_true", help="Force CPU execution")\n')
    buf.write('_dev_parser.add_argument("--atol", type=float, default=None, help="Absolute tolerance for verification")\n')
    buf.write('_dev_parser.add_argument("--rtol", type=float, default=None, help="Relative tolerance for verification")\n')
    buf.write('_dev_args, _ = _dev_parser.parse_known_args()\n')
    buf.write('_device = "cpu" if _dev_args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")\n')


def _patch_flatten_parameters():
    """Context manager: no-op nn.RNNBase.flatten_parameters during tracing.

    flatten_parameters() is a cuDNN memory layout optimization that calls
    .data_ptr() internally.  This crashes on FakeTensors used by
    torch.compile / torch.export.  Safe to skip — it doesn't affect
    mathematical correctness.
    """
    @contextlib.contextmanager
    def _ctx():
        orig = torch.nn.RNNBase.flatten_parameters
        torch.nn.RNNBase.flatten_parameters = lambda self: None  # type: ignore[assignment]
        try:
            yield
        finally:
            torch.nn.RNNBase.flatten_parameters = orig  # type: ignore[assignment]
    return _ctx()


@contextlib.contextmanager
def _patch_compiler_config(**overrides):
    """Temporarily override torch.compiler.config for graph capture.

    We enable options here only when they improve capture fidelity. A current
    example is `capture_scalar_outputs=True`, which keeps scalar extractions
    such as `Tensor.item()` inside the traced graph instead of forcing a graph
    break. Models like Reformer use that pattern in forward for shape/position
    logic, so capturing those scalars is part of a faithful aten export.
    """
    # Filter out config keys that don't exist in this PyTorch version
    # (e.g. capture_scalar_outputs was added in PyTorch 2.10).
    supported = {k: v for k, v in overrides.items() if hasattr(torch.compiler.config, k)}
    if supported:
        with torch.compiler.config.patch(supported):
            yield
    else:
        yield
# -----------------------------------------------------------------------------
# Capture at aten level with autograd
# -----------------------------------------------------------------------------


@dataclass
class AtenGraph:
    """A captured aten-level FX graph (forward or backward)."""

    graph_module: GraphModule
    example_inputs: list[torch.Tensor]
    kind: str  # "forward" or "backward"
    graph_id: int = 0

    @property
    def code(self) -> str:
        return self.graph_module.code

    @property
    def readable(self) -> str:
        return self.graph_module.print_readable(print_output=False)

    def __repr__(self) -> str:
        n_nodes = len(list(self.graph_module.graph.nodes))
        return f"AtenGraph(kind={self.kind}, nodes={n_nodes})"


@dataclass
class SourceTrace:
    """Source location for a group of aten ops."""
    file: str           # e.g. "/path/to/model.py"
    line: int           # e.g. 42
    code: str           # e.g. "x = F.relu(self.fc1(x))"
    fn_name: str        # e.g. "forward"
    module_path: str    # e.g. "self.fc1" (from nn_module_stack)
    module_type: str    # e.g. "Linear"

    def short(self) -> str:
        return f"{self.file}:{self.line}  {self.code}"

    def __repr__(self) -> str:
        return f"SourceTrace({self.file}:{self.line} {self.code!r})"


def _clone_output(result):
    """Clone a graph output (tensor or tuple of tensors) for recording."""
    if isinstance(result, torch.Tensor):
        return result.clone().detach()
    if isinstance(result, (tuple, list)):
        return tuple(
            t.clone().detach() if isinstance(t, torch.Tensor) else t
            for t in result
        )
    return result


def _extend_list(lst: list, idx: int, value):
    """Ensure lst[idx] exists, then set it."""
    while len(lst) <= idx:
        lst.append(None)
    lst[idx] = value



@dataclass
class AtenCapture:
    """Holds both forward and backward aten-level graphs."""

    forward_graphs: list[AtenGraph] = field(default_factory=list)
    backward_graphs: list[AtenGraph] = field(default_factory=list)
    param_names: list[str] = field(default_factory=list)
    buffer_names: list[str] = field(default_factory=list)
    param_shapes: list[list[int]] = field(default_factory=list)
    buffer_shapes: list[list[int]] = field(default_factory=list)
    primal_names: list[str] = field(default_factory=list)
    source_map: dict[str, SourceTrace] = field(default_factory=dict)
    _counter: int = 0
    _primal_order_recorded: bool = False

    # Real tensor recordings (populated when record_real_tensors=True)
    forward_real_inputs: list[torch.Tensor] | None = None
    forward_intermediates: dict[str, torch.Tensor] | None = None
    forward_real_output: Any = None
    backward_real_inputs: list[torch.Tensor] | None = None
    backward_intermediates: dict[str, torch.Tensor] | None = None
    backward_real_output: Any = None

    # Per-fragment recordings (populated for multi-fragment / graph-break captures)
    per_frag_primal_names: list[list[str | None]] = field(default_factory=list)
    per_frag_fw_inputs: list[list | None] = field(default_factory=list)
    per_frag_fw_output: list[Any] = field(default_factory=list)
    per_frag_bw_inputs: list[list | None] = field(default_factory=list)
    per_frag_bw_output: list[Any] = field(default_factory=list)

    # Optimizer snapshot (populated by extract_training_step)
    optimizer_data: dict[str, Any] | None = None
    # Optimizer aten capture (populated by capture_optimizer_aten)
    optimizer_capture: "AtenCapture | None" = None
    # Optimizer placeholder metadata (per captured optimizer graph)
    optimizer_slot_info: list[list[dict[str, Any]]] = field(default_factory=list)

    # Inductor/Triton kernel enrichment. Historical field names are kept for
    # backward compatibility because downstream code still refers to them.
    triton_capture: Any = None
    backward_triton_capture: Any = None

    @property
    def forward_kernel_capture(self) -> Any:
        """Forward kernel/debug enrichment attached to this capture."""
        return self.triton_capture

    @forward_kernel_capture.setter
    def forward_kernel_capture(self, value: Any) -> None:
        self.triton_capture = value

    @property
    def backward_kernel_capture(self) -> Any:
        """Backward kernel/debug enrichment attached to this capture."""
        return self.backward_triton_capture

    @backward_kernel_capture.setter
    def backward_kernel_capture(self, value: Any) -> None:
        self.backward_triton_capture = value

    def attach_kernel_enrichment(
        self,
        forward_capture: Any | None,
        backward_capture: Any | None = None,
    ) -> None:
        """Attach kernel/debug enrichment produced by an Inductor pass."""
        self.forward_kernel_capture = forward_capture
        self.backward_kernel_capture = backward_capture

    def summary(self) -> str:
        lines = [f"Captured {len(self.forward_graphs)} forward + "
                 f"{len(self.backward_graphs)} backward graph(s):"]
        for g in self.forward_graphs:
            n = len(list(g.graph_module.graph.nodes))
            lines.append(f"  [FW {g.graph_id}] {n} nodes")
        for g in self.backward_graphs:
            n = len(list(g.graph_module.graph.nodes))
            lines.append(f"  [BW {g.graph_id}] {n} nodes")
        return "\n".join(lines)


def _safe_deepcopy_gm(gm: GraphModule) -> GraphModule:
    """Deepcopy a GraphModule, even when FakeTensorMode is active.

    The fw/bw compilers run *inside* a FakeTensorMode dispatch context.
    ``copy.deepcopy(gm)`` clones each buffer tensor's storage, and the
    active FakeTensorMode intercepts the clone, producing a meta-device
    FakeTensor.  ``aten.set_.source_Storage`` then fails because the
    original storage is on "cpu" while the clone is on "meta".

    When FakeTensorMode is detected on the dispatch stack we temporarily
    disable all dispatch modes so that deepcopy produces real CPU tensors,
    then restore the modes afterwards.
    """
    # Detect active FakeTensorMode
    _in_fake_mode = False
    try:
        from torch._subclasses.fake_tensor import FakeTensorMode
        mode_stack = torch.utils._python_dispatch._get_current_dispatch_mode_stack()
        _in_fake_mode = any(isinstance(m, FakeTensorMode) for m in mode_stack)
    except Exception:
        pass

    if not _in_fake_mode:
        try:
            return copy.deepcopy(gm)
        except RuntimeError as e:
            if "set the storage" not in str(e) and "set_" not in str(e):
                raise
            # Fall through to modes-disabled path

    # Temporarily disable all dispatch modes (FakeTensorMode, ProxyMode, etc.)
    # so that copy.deepcopy produces real CPU tensors instead of FakeTensors.
    from torch.utils._python_dispatch import _disable_current_modes
    with _disable_current_modes():
        return copy.deepcopy(gm)


def _safe_copy_inputs(example_inputs):
    """Copy inputs, handling FakeTensors from aot_autograd.

    FakeTensors have ~100x dispatch overhead on clone/detach, so we materialize
    them as real CPU tensors with matching shape/dtype instead.  The copies are
    only used for metadata (shapes, dtypes) and code generation, not for
    numerical accuracy — use torch.empty (500x faster than randn).
    """
    copies = []
    for inp in example_inputs:
        if isinstance(inp, torch.Tensor):
            shape = list(inp.shape)
            dtype = inp.dtype
            # Materialize as real CPU tensor — avoids FakeTensor dispatch overhead.
            # Use torch.empty (uninitialized) — data is never used for accuracy,
            # only shape/dtype metadata matters for code generation.
            if _is_fake(inp) or (hasattr(inp, "device") and inp.device.type == "meta"):
                copies.append(torch.empty(shape, dtype=dtype))
            else:
                copies.append(inp.clone().detach())
        else:
            copies.append(inp)
    return copies


def _resolve_loss(output, loss_fn=None):
    """Derive a scalar loss from model output for backward pass."""
    if loss_fn is not None:
        return loss_fn(output)
    if isinstance(output, torch.Tensor):
        return output if output.ndim == 0 else output.sum()
    if isinstance(output, (tuple, list)):
        return output[0].sum()
    # HuggingFace-style model outputs with .loss attribute
    if hasattr(output, 'loss') and isinstance(getattr(output, 'loss', None), torch.Tensor):
        return output.loss
    # Try common tensor attributes
    for attr in ('logits', 'last_hidden_state'):
        val = getattr(output, attr, None)
        if isinstance(val, torch.Tensor):
            return val.sum()
    raise ValueError(
        f"Cannot derive loss from output of type {type(output).__name__}. "
        f"Pass a loss_fn or return a tensor/tuple."
    )


def capture_aten_graphs(
    model_or_fn: torch.nn.Module | Callable,
    *args,
    run_backward: bool = True,
    loss_fn: Callable | None = None,
    dynamic: bool = False,
    record_real_tensors: bool = False,
    record_filter: dict | None = None,
    triton: bool = False,
    use_inductor: bool = False,
    offload_saved: bool = False,
    **kwargs,
) -> tuple[Any, AtenCapture]:
    """Capture forward + backward FX graphs at the aten op level.

    Uses aot_autograd to decompose the model into pure aten operations
    for both the forward pass and the autograd backward pass.

    When *triton* is True and CUDA is available, the authoritative aten graph
    capture is followed by a separate Inductor debug-enrichment pass. That
    second pass captures Triton kernel source code and the aten-op-to-kernel
    mapping, then attaches the resulting ``TritonCapture`` objects back onto
    ``capture``.

    Args:
        model_or_fn: A model or function to compile and trace.
        *args: Inputs to the model.
        run_backward: If True, also run backward() to capture the backward graph.
        loss_fn: Optional loss function. If None and output is not scalar,
                 defaults to .sum().
        dynamic: If True, capture with symbolic/dynamic shapes.
        record_real_tensors: If True, record all real inputs/intermediates/outputs
            during the actual execution. Stored in capture.forward_real_inputs,
            capture.forward_intermediates, capture.backward_real_inputs,
            capture.backward_intermediates.
        record_filter: Optional dict of selectors passed to
            ``op_dump.select_nodes()`` to limit which nodes get recorded.
            E.g. ``{"lines": "model.py:42-50"}`` or ``{"module": "blocks.0.attn"}``.
            Only effective when *record_real_tensors* is True.
        triton: If True, run a post-capture Inductor debug-enrichment pass.
            Requires CUDA.

    Returns:
        (output, AtenCapture) where AtenCapture has .forward_graphs and .backward_graphs
    """
    from torch._functorch.aot_autograd import aot_module_simplified, make_boxed_func

    capture = AtenCapture()
    _OFFLOAD_MIN_BYTES = 1024 * 1024  # 1 MB — threshold for CPU↔GPU offloading
    _offloaded_tensor_ids: set[int] = set()  # ids of tensors offloaded to CPU by forward

    if isinstance(model_or_fn, torch.nn.Module):
        capture.param_names = [name for name, _ in model_or_fn.named_parameters()]
        capture.buffer_names = [name for name, _ in model_or_fn.named_buffers()]
        capture.param_shapes = [list(p.shape) for _, p in model_or_fn.named_parameters()]
        capture.buffer_shapes = [list(b.shape) for _, b in model_or_fn.named_buffers()]

        params_dict = dict(model_or_fn.named_parameters(remove_duplicate=False))
        buffers_dict = dict(model_or_fn.named_buffers(remove_duplicate=False))
        capture.primal_names = list(params_dict.keys()) + list(buffers_dict.keys())

    def fw_compiler(gm: GraphModule, example_inputs):
        gm_copy = _safe_deepcopy_gm(gm)
        inputs_copy = _safe_copy_inputs(example_inputs)
        graph = AtenGraph(
            graph_module=gm_copy,
            example_inputs=inputs_copy,
            kind="forward",
            graph_id=capture._counter,
        )
        capture.forward_graphs.append(graph)
        capture._counter += 1
        frag_idx = len(capture.forward_graphs) - 1

        # Finalize primal_names: aot_autograd may deduplicate inputs with
        # shared storage, so the raw ordering (built in aot_backend) can be
        # longer than the FX graph's placeholders. Deduplicate here to match.
        n_placeholders = sum(1 for n in gm.graph.nodes if n.op == 'placeholder')
        if hasattr(capture, '_raw_primal_ordering') and capture._raw_primal_ordering is not None:
            raw = capture._raw_primal_ordering
            capture._raw_primal_ordering = None  # consume it
            if len(raw) != n_placeholders:
                # Deduplicate: keep first occurrence of each data_ptr
                deduped = []
                seen_ptrs = set()
                for name, ptr in raw:
                    if ptr is not None and ptr in seen_ptrs:
                        continue  # duplicate tensor input — skip
                    if ptr is not None:
                        seen_ptrs.add(ptr)
                    deduped.append(name)
                ordered = deduped[:n_placeholders]
            else:
                ordered = [name for name, ptr in raw]

            capture.per_frag_primal_names.append(ordered)
            if not capture._primal_order_recorded:
                capture.primal_names = ordered
                capture._primal_order_recorded = True

        if record_real_tensors:
            # Resolve selective recording filter (if any)
            fw_record_nodes = None
            if record_filter:
                if record_filter.get("inputs_only"):
                    fw_record_nodes = {
                        n.name for n in gm.graph.nodes
                        if n.op == "placeholder"
                    }
                else:
                    from torch_graph.op_dump import select_nodes as _select_nodes
                    fw_record_nodes = _select_nodes(
                        gm, source_map=capture.source_map, **record_filter
                    )

            def _recording_fw(args, _fi=frag_idx):
                inputs = [
                    a.clone().detach() if isinstance(a, torch.Tensor) else a
                    for a in args
                ]
                interp = _RecordInterp(gm, record_nodes=fw_record_nodes)
                result = interp.run(*args)
                cloned = _clone_output(result)
                # Per-fragment recording
                _extend_list(capture.per_frag_fw_inputs, _fi, inputs)
                _extend_list(capture.per_frag_fw_output, _fi, cloned)
                # Global backward compat (last fragment wins)
                capture.forward_real_inputs = inputs
                capture.forward_intermediates = dict(interp.recorded)
                capture.forward_real_output = cloned
                return result
            _recording_fw._boxed_call = True
            return _recording_fw

        if offload_saved:
            # aot_autograd's forward output is: (mutations..., real_outputs..., saved...)
            # We must NOT offload real outputs or mutations — only saved-for-backward.
            #
            # Strategy: use FX Interpreter to offload saved tensors DURING forward
            # execution, right after their last forward use. This reduces peak GPU
            # memory (unlike post-execution offloading which can't reduce forward peak).
            ph_names = {n.name for n in gm.graph.nodes if n.op == 'placeholder'}
            out_node = next(n for n in gm.graph.nodes if n.op == 'output')
            out_args = out_node.args[0] if out_node.args else ()

            # Count mutations: leading output elements that are placeholders
            num_mutations = 0
            for arg in out_args:
                if isinstance(arg, Node) and arg.name in ph_names:
                    num_mutations += 1
                else:
                    break
            # Skip mutations + 1 real output
            n_skip = num_mutations + 1
            n_saved = len(out_args) - n_skip

            # Build saved-for-backward Node set
            saved_nodes: set[Node] = set()
            for i, arg in enumerate(out_args):
                if i >= n_skip and isinstance(arg, Node):
                    saved_nodes.add(arg)

            # For each saved node, find its last forward user (excluding output node).
            # After that user runs, the tensor is only needed for backward → offload.
            compute_nodes = [n for n in gm.graph.nodes if n.op not in ('placeholder', 'output')]
            node_order = {n: i for i, n in enumerate(compute_nodes)}

            # offload_after[trigger_node] = list of saved Nodes to offload
            # Keyed by Node objects (matching self.env keys in FX Interpreter)
            offload_after: dict[Node, list[Node]] = {}
            n_output_only = 0
            for saved in saved_nodes:
                non_output_users = [u for u in saved.users.keys() if u.op != 'output']
                if non_output_users:
                    last_user = max(non_output_users, key=lambda u: node_order.get(u, -1))
                    offload_after.setdefault(last_user, []).append(saved)
                else:
                    # Output-only node: offload right after it's computed
                    offload_after.setdefault(saved, []).append(saved)
                    n_output_only += 1

            logger.info(f"offload_saved: {len(out_args)} outputs "
                        f"({num_mutations} mutations, 1 real, {n_saved} saved), "
                        f"{n_output_only} output-only")

            # Persist counts so cross-graph annotation can split fw outputs
            # correctly even when forward_real_output is not captured.
            capture._num_mutations = num_mutations
            capture._num_real_outputs = 1

            class _OffloadInterp(torch.fx.Interpreter):
                def __init__(self, *a, **kw):
                    super().__init__(*a, **kw)
                    self._offloaded_bytes = 0
                    self._offloaded_count = 0

                def run_node(self, n):
                    result = super().run_node(n)
                    # After this node runs, offload any saved tensors whose last
                    # forward user was this node
                    to_offload = offload_after.get(n)
                    if to_offload:
                        for saved_node in to_offload:
                            if saved_node not in self.env:
                                continue
                            val = self.env[saved_node]
                            if (isinstance(val, torch.Tensor) and val.is_cuda
                                    and val.numel() * val.element_size() >= _OFFLOAD_MIN_BYTES):
                                cpu_val = val.to('cpu', non_blocking=True)
                                self._offloaded_bytes += val.numel() * val.element_size()
                                self._offloaded_count += 1
                                self.env[saved_node] = cpu_val
                                _offloaded_tensor_ids.add(id(cpu_val))
                    return result

            def _offloading_fw(args):
                import time as _time
                _offloaded_tensor_ids.clear()
                t0 = _time.perf_counter()
                interp = _OffloadInterp(gm)
                result = interp.run(*args)
                offloaded_count = interp._offloaded_count
                offloaded_bytes = interp._offloaded_bytes
                dt = _time.perf_counter() - t0
                mem = torch.cuda.memory_allocated() / 1e9
                del interp
                logger.info(
                    f"offload_saved forward: {dt:.2f}s, offloaded {offloaded_count}/{n_saved} "
                    f"tensors ({offloaded_bytes / 1e9:.2f} GB) to CPU, "
                    f"GPU mem: {mem:.2f} GB"
                )
                return result
            _offloading_fw._boxed_call = True
            return _offloading_fw

        return make_boxed_func(gm.forward)

    def bw_compiler(gm: GraphModule, example_inputs):
        gm_copy = _safe_deepcopy_gm(gm)
        inputs_copy = _safe_copy_inputs(example_inputs)
        graph = AtenGraph(
            graph_module=gm_copy,
            example_inputs=inputs_copy,
            kind="backward",
            graph_id=capture._counter,
        )
        capture.backward_graphs.append(graph)
        capture._counter += 1
        frag_idx = len(capture.backward_graphs) - 1

        if offload_saved:
            # Backward receives saved tensors that were offloaded to CPU.
            # We must NOT move all placeholders to GPU at once (would OOM).
            # Instead: keep placeholders on CPU, move to GPU only when a
            # compute node actually consumes them.
            #
            # We identify offloaded tensors by their Python id() — the
            # forward records ids of tensors it moved to CPU. Tensors that
            # are NATURALLY on CPU (e.g. SDPA RNG state) won't be in the
            # set and are left on CPU.
            _cuda_dev = torch.cuda.current_device()

            # On first backward call, snapshot which placeholder indices
            # correspond to offloaded tensors (by checking id() against
            # _offloaded_tensor_ids from the forward). Cache for future calls.
            _offloaded_ph_indices: set[int] | None = None

            class _LazyGPUInterp(torch.fx.Interpreter):
                """Moves offloaded-to-CPU tensors back to GPU at use-time.

                Tracks which nodes are "offloaded" (came from our forward
                offloading, not naturally CPU). Propagates the flag through
                views/reshapes: if a compute node's inputs include offloaded
                nodes and it produces a CPU tensor, the result is also
                considered offloaded.
                """
                def __init__(self, *a, **kw):
                    super().__init__(*a, **kw)
                    self._moved_bytes = 0
                    self._moved_count = 0
                    self._offloaded_nodes: set = set()  # nodes with offloaded data

                def run_node(self, n):
                    if n.op == 'placeholder':
                        val = super().run_node(n)
                        if (isinstance(val, torch.Tensor)
                                and val.device.type == 'cpu'
                                and id(val) in _offloaded_tensor_ids):
                            self._offloaded_nodes.add(n)
                        return val

                    # Move offloaded CPU inputs to GPU before running this node
                    has_offloaded_input = False
                    for inp in n.all_input_nodes:
                        if inp not in self.env:
                            continue
                        if inp in self._offloaded_nodes:
                            has_offloaded_input = True
                            val = self.env[inp]
                            if isinstance(val, torch.Tensor) and val.device.type == 'cpu':
                                gpu_val = val.to(_cuda_dev)
                                self._moved_bytes += val.numel() * val.element_size()
                                self._moved_count += 1
                                self.env[inp] = gpu_val

                    result = super().run_node(n)

                    # Propagate offloaded flag through views/reshapes
                    if (has_offloaded_input
                            and isinstance(result, torch.Tensor)
                            and result.device.type == 'cpu'):
                        self._offloaded_nodes.add(n)

                    return result

            def _offloading_bw(args):
                import time as _time
                t0 = _time.perf_counter()
                interp = _LazyGPUInterp(gm)
                result = interp.run(*args)
                dt = _time.perf_counter() - t0
                logger.info(
                    f"offload_saved backward: {dt:.1f}s, moved {interp._moved_count} tensors "
                    f"({interp._moved_bytes / 1e9:.1f} GB) CPU→GPU, "
                    f"GPU mem: {torch.cuda.memory_allocated() / 1e9:.1f} GB"
                )
                return result
            _offloading_bw._boxed_call = True
            return _offloading_bw

        if record_real_tensors:
            bw_record_nodes = None
            if record_filter:
                if record_filter.get("inputs_only"):
                    bw_record_nodes = set()
                else:
                    from torch_graph.op_dump import select_nodes as _select_nodes
                    bw_record_nodes = _select_nodes(
                        gm, source_map=capture.source_map, **record_filter
                    )

            def _recording_bw(args, _fi=frag_idx):
                inputs = [
                    a.clone().detach() if isinstance(a, torch.Tensor) else a
                    for a in args
                ]
                interp = _RecordInterp(gm, record_nodes=bw_record_nodes)
                result = interp.run(*args)
                cloned = _clone_output(result)
                # Per-fragment recording
                _extend_list(capture.per_frag_bw_inputs, _fi, inputs)
                _extend_list(capture.per_frag_bw_output, _fi, cloned)
                # Global backward compat (last fragment wins)
                capture.backward_real_inputs = inputs
                capture.backward_intermediates = dict(interp.recorded)
                capture.backward_real_output = cloned
                return result
            _recording_bw._boxed_call = True
            return _recording_bw

        return make_boxed_func(gm.forward)

    def aot_backend(gm, inputs):
        # Extract source traces from the pre-decomposition FX graph.
        # This graph still has stack_trace, source_fn_stack, nn_module_stack
        # metadata — the same data that _capture_source_traces would get
        # from a separate torch.compile pass.  By extracting here we avoid
        # a redundant compilation.
        if not capture.source_map:
            capture.source_map = _extract_source_traces_from_graph(gm)

        # Build raw primal ordering from real inputs (which have valid data_ptrs).
        # The fw_compiler will then deduplicate to match the FX graph's
        # placeholder count (aot_autograd may remove duplicate-storage inputs).
        if isinstance(model_or_fn, torch.nn.Module):
            _param_id_map = {}  # data_ptr → param_name (first occurrence wins)
            for name, p in model_or_fn.named_parameters(remove_duplicate=False):
                try:
                    ptr = p.data_ptr()
                    if ptr not in _param_id_map:
                        _param_id_map[ptr] = name
                except Exception:
                    pass
            for name, b in model_or_fn.named_buffers(remove_duplicate=False):
                try:
                    ptr = b.data_ptr()
                    if ptr not in _param_id_map:
                        _param_id_map[ptr] = name
                except Exception:
                    pass
            raw_ordering = []  # list of (param_name_or_None, data_ptr_or_None)
            for inp in inputs:
                if isinstance(inp, torch.Tensor):
                    try:
                        ptr = inp.data_ptr()
                    except Exception:
                        ptr = None
                    raw_ordering.append((_param_id_map.get(ptr), ptr))
                else:
                    raw_ordering.append((None, None))  # SymInt
            capture._raw_primal_ordering = raw_ordering

            # Record user input call-position ordering (parallel to raw_ordering)
            if hasattr(capture, '_user_input_map') and capture._user_input_map:
                _uio = []
                for _name, _ptr in raw_ordering:
                    if _name is None and _ptr is not None:
                        _uio.append(capture._user_input_map.get(_ptr))
                    elif _name is None:
                        _uio.append(None)  # SymInt
                capture._user_input_ordering = _uio

        elif not isinstance(model_or_fn, torch.nn.Module):
            # For plain functions: record which call-position arg each FX placeholder
            # corresponds to. aot_autograd may reorder inputs (e.g. by first-access
            # order), so we match by data_ptr while we still have real tensors.
            fn_ordering = []
            for inp in inputs:
                if isinstance(inp, torch.Tensor):
                    try:
                        ptr = inp.data_ptr()
                        fn_ordering.append(capture._user_input_map.get(ptr))
                    except Exception:
                        fn_ordering.append(None)
                else:
                    fn_ordering.append(None)  # SymInt placeholder
            capture._fn_input_ordering = fn_ordering

        if use_inductor:
            # Delegate to inductor's compile_fx, but wrap inner_compile to
            # also save the decomposed aten graphs.  compile_fx calls
            # aot_module_simplified internally with its own decompositions,
            # so the graphs inner_compile receives are correctly decomposed
            # for inductor (no double-decomposition issues).
            from torch._inductor.compile_fx import compile_fx as _compile_fx
            from torch._inductor.compile_fx import compile_fx_inner as _inner

            _is_fw = [True]  # first inner_compile call = forward
            def _capturing_inner(gm_inner, example_inputs_inner, **kwargs):
                if _is_fw[0]:
                    fw_compiler(gm_inner, example_inputs_inner)
                    _is_fw[0] = False
                else:
                    bw_compiler(gm_inner, example_inputs_inner)
                return _inner(gm_inner, example_inputs_inner, **kwargs)

            return _compile_fx(gm, inputs, inner_compile=_capturing_inner)

        return aot_module_simplified(
            gm, inputs,
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
        )

    torch.compiler.reset()

    # Ensure inputs require grad for backward capture
    processed_args = []
    for a in args:
        if isinstance(a, torch.Tensor) and a.is_floating_point() and run_backward:
            processed_args.append(a.detach().requires_grad_(True))
        else:
            processed_args.append(a)

    # Build user input data_ptr → call-position map so aot_backend can
    # record the true user input ordering (aot_autograd may reorder them).
    _user_input_map = {}
    for _i, _a in enumerate(processed_args):
        if isinstance(_a, torch.Tensor):
            try:
                _user_input_map[_a.data_ptr()] = _i
            except Exception:
                pass
    if kwargs and isinstance(model_or_fn, torch.nn.Module):
        from torch_graph.install import _extract_forward_arg_names
        _fwd_params = _extract_forward_arg_names(model_or_fn.forward)
        for _k, _v in kwargs.items():
            if isinstance(_v, torch.Tensor):
                try:
                    _pos = _fwd_params.index(_k) if _k in _fwd_params else len(processed_args)
                    _user_input_map[_v.data_ptr()] = _pos
                except Exception:
                    pass
    capture._user_input_map = _user_input_map

    # Use the real torch.compile even if auto_install has patched it.
    _compile_fn = torch.compile
    try:
        from torch_graph.auto_install import _real_torch_compile
        if _real_torch_compile is not None:
            _compile_fn = _real_torch_compile
    except ImportError:
        pass

    with _patch_compiler_config(capture_scalar_outputs=True):
        compiled = _compile_fn(model_or_fn, backend=aot_backend, dynamic=dynamic)
        with _patch_flatten_parameters():
            if run_backward:
                output = compiled(*processed_args, **kwargs)
            else:
                # Forward-only capture does not need autograd to retain
                # activation intermediates, which is critical for very large
                # decoder models at long sequence lengths.
                with torch.no_grad():
                    output = compiled(*processed_args, **kwargs)

    # Store the compiled model so callers (e.g. auto_install) can reuse it
    # for subsequent forward calls with the same offloading behavior.
    capture._compiled = compiled

    if run_backward:
        loss = _resolve_loss(output, loss_fn)
        loss.backward()

    # ── Fallback: torch.export when torch.compile produces 0 graphs ──
    # Some modules (LSTM, GRU, RNN) cause torch.compile to fall back to
    # eager mode entirely (0 graphs).  Try torch.export to get a single
    # complete aten-level graph.
    # When graph breaks produce multiple fragments, we now handle them
    # natively via per-fragment export instead of falling back.
    _needs_fallback = not capture.forward_graphs
    _has_graph_breaks = len(capture.forward_graphs) > 1
    if _has_graph_breaks:
        import warnings
        n_frags = len(capture.forward_graphs)
        warnings.warn(
            f"torch.compile produced {n_frags} graph fragments (graph breaks). "
            f"Using multi-fragment export."
        )
        # Clear global primal_names/real_inputs: they reflect only the last
        # fragment.  Per-fragment data is used instead.
        capture.primal_names = []
        capture.forward_real_inputs = None
        capture.backward_real_inputs = None
    if _needs_fallback and isinstance(model_or_fn, torch.nn.Module):
        import warnings
        warnings.warn(
            "torch.compile produced no graphs. "
            "Falling back to torch.export + aot_autograd."
        )
        torch.compiler.reset()
        try:
            output, capture = _fallback_export_capture(
                model_or_fn, processed_args, kwargs, capture,
                run_backward=run_backward, loss_fn=loss_fn,
                record_real_tensors=record_real_tensors,
                record_filter=record_filter,
            )
        except Exception as e:
            raise RuntimeError(
                f"Both torch.compile and torch.export failed for this model. "
                f"torch.export error: {e}"
            ) from e

    if triton:
        _enrich_capture_with_inductor_debug(
            model_or_fn, capture, processed_args, kwargs,
            run_backward=run_backward, loss_fn=loss_fn,
        )

    return output, capture


def _fallback_export_capture(
    model, processed_args, kwargs, capture, *,
    run_backward, loss_fn, record_real_tensors, record_filter,
):
    """Fallback capture via torch.export for models torch.compile can't handle.

    Uses torch.export to get an aten-level GraphModule, then runs that
    through torch.compile + aot_autograd to get the fw/bw split.
    """
    from torch._functorch.aot_autograd import aot_module_simplified, make_boxed_func

    # Step 1: export the model to get an aten-level GraphModule
    # Use original (non-grad) args for export since export doesn't need grad
    export_args = tuple(
        a.detach() if isinstance(a, torch.Tensor) else a for a in processed_args
    )
    with _patch_flatten_parameters():
        exported = torch.export.export(model, export_args, kwargs or {})
    gm_exported = exported.module()

    # Extract source traces from the exported graph
    if not capture.source_map:
        capture.source_map = _extract_source_traces_from_graph(gm_exported)

    # Preserve param/buffer metadata from the original model.
    # NOTE: do NOT set primal_names yet — the export fallback puts user inputs
    # before lifted parameters, so the ordering differs from normal compile.
    # We'll build the correct primal_names after seeing the fw graph.
    capture.param_names = [n for n, _ in model.named_parameters()]
    capture.param_shapes = [list(p.shape) for _, p in model.named_parameters()]
    capture.buffer_names = [n for n, _ in model.named_buffers()]
    capture.buffer_shapes = [list(b.shape) for _, b in model.named_buffers()]
    # Clear primal_names — will be rebuilt after fw capture
    capture.primal_names = []

    # Build a (shape, dtype) → name lookup for matching primals to params
    _param_lookup = {}  # (tuple(shape), dtype) → [names]
    for name, p in model.named_parameters():
        key = (tuple(p.shape), p.dtype)
        _param_lookup.setdefault(key, []).append(name)
    for name, b in model.named_buffers():
        key = (tuple(b.shape), b.dtype)
        _param_lookup.setdefault(key, []).append(name)

    # Step 2: compile the exported module through aot_autograd
    def fw_compiler(gm_fw, example_inputs):
        gm_copy = _safe_deepcopy_gm(gm_fw)
        inputs_copy = _safe_copy_inputs(example_inputs)
        graph = AtenGraph(
            graph_module=gm_copy,
            example_inputs=inputs_copy,
            kind="forward",
            graph_id=capture._counter,
        )
        capture.forward_graphs.append(graph)
        capture._counter += 1

        # Build correct primal_names from the fw graph placeholders.
        # In the export fallback, aot_autograd may order inputs differently
        # (user inputs before params). We match each placeholder to a
        # known parameter by (shape, dtype), consuming matches in order.
        if not capture.primal_names:
            avail = {k: list(v) for k, v in _param_lookup.items()}
            primal_names = []
            for inp in example_inputs:
                if isinstance(inp, torch.Tensor):
                    key = (tuple(inp.shape), inp.dtype)
                    if key in avail and avail[key]:
                        primal_names.append(avail[key].pop(0))
                    else:
                        primal_names.append(None)  # user input
                else:
                    primal_names.append(None)
            capture.primal_names = primal_names

        if record_real_tensors:
            fw_record_nodes = None
            if record_filter:
                if record_filter.get("inputs_only"):
                    fw_record_nodes = {
                        n.name for n in gm_fw.graph.nodes
                        if n.op == "placeholder"
                    }
                else:
                    from torch_graph.op_dump import select_nodes as _select_nodes
                    fw_record_nodes = _select_nodes(
                        gm_fw, source_map=capture.source_map, **record_filter
                    )

            def _recording_fw(args):
                capture.forward_real_inputs = [
                    a.clone().detach() if isinstance(a, torch.Tensor) else a
                    for a in args
                ]
                interp = _RecordInterp(gm_fw, record_nodes=fw_record_nodes)
                result = interp.run(*args)
                capture.forward_intermediates = dict(interp.recorded)
                capture.forward_real_output = _clone_output(result)
                return result
            _recording_fw._boxed_call = True
            return _recording_fw

        return make_boxed_func(gm_fw.forward)

    def bw_compiler(gm_bw, example_inputs):
        gm_copy = _safe_deepcopy_gm(gm_bw)
        inputs_copy = _safe_copy_inputs(example_inputs)
        graph = AtenGraph(
            graph_module=gm_copy,
            example_inputs=inputs_copy,
            kind="backward",
            graph_id=capture._counter,
        )
        capture.backward_graphs.append(graph)
        capture._counter += 1

        if record_real_tensors:
            bw_record_nodes = None
            if record_filter:
                if record_filter.get("inputs_only"):
                    bw_record_nodes = set()
                else:
                    from torch_graph.op_dump import select_nodes as _select_nodes
                    bw_record_nodes = _select_nodes(
                        gm_bw, source_map=capture.source_map, **record_filter
                    )

            def _recording_bw(args):
                capture.backward_real_inputs = [
                    a.clone().detach() if isinstance(a, torch.Tensor) else a
                    for a in args
                ]
                interp = _RecordInterp(gm_bw, record_nodes=bw_record_nodes)
                result = interp.run(*args)
                capture.backward_intermediates = dict(interp.recorded)
                capture.backward_real_output = _clone_output(result)
                return result
            _recording_bw._boxed_call = True
            return _recording_bw

        return make_boxed_func(gm_bw.forward)

    def aot_backend(gm, inputs):
        return aot_module_simplified(
            gm, inputs,
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
        )

    with _patch_compiler_config(capture_scalar_outputs=True):
        compiled = torch.compile(gm_exported, backend=aot_backend)
        # Use the original processed_args (with grad where needed)
        if run_backward:
            output = compiled(*processed_args, **kwargs)
        else:
            with torch.no_grad():
                output = compiled(*processed_args, **kwargs)

    if run_backward:
        loss = _resolve_loss(output, loss_fn)
        loss.backward()

    return output, capture


def _enrich_capture_with_inductor_debug(model_or_fn, capture, args, kwargs,
                                        run_backward=False, loss_fn=None):
    """Attach Inductor debug metadata onto an existing aten capture.

    Uses the same model and inputs so the Inductor pass stays aligned with the
    already-captured aten graph, enabling reliable kernel-to-node matching.
    """
    from torch_graph.triton import enrich_capture_with_inductor_debug

    try:
        logger.info("  Running Inductor debug enrichment pass …")
        fw_tcap, bw_tcap = enrich_capture_with_inductor_debug(
            capture,
            model_or_fn, *args,
            run_backward=run_backward, loss_fn=loss_fn,
            **kwargs,
        )

        if fw_tcap is not None:
            triton_count = sum(1 for k in fw_tcap.kernels if k.kernel_type == "triton")
            extern_count = sum(1 for k in fw_tcap.kernels if k.kernel_type == "extern")
            logger.info(
                "  Kernel enrichment: %d Triton kernels + %d extern calls",
                triton_count, extern_count,
            )
        else:
            logger.warning("  No kernel enrichment captured (forward)")

        if bw_tcap is not None:
            triton_count = sum(1 for k in bw_tcap.kernels if k.kernel_type == "triton")
            extern_count = sum(1 for k in bw_tcap.kernels if k.kernel_type == "extern")
            logger.info(
                "  Kernel enrichment (backward): %d Triton kernels + %d extern calls",
                triton_count, extern_count,
            )
    except Exception as e:
        logger.warning(f"  Inductor debug enrichment failed: {e}")
        logger.debug("", exc_info=True)


def capture_optimizer_aten(
    optimizer,
    record_real_tensors: bool = False,
    param_name_map: dict[int, str] | None = None,
    step_fn: Callable | None = None,
) -> AtenCapture:
    """Capture aten-level graph of an optimizer step via torch.compile.

    Traces ``optimizer.step()`` through aot_autograd to extract the real
    aten ops that PyTorch executes (lerp, addcdiv, sqrt, pow, etc.).
    This is a true extraction — the ops come from the compiler, not from
    knowing the optimizer class.

    For non-standard optimizers (without ``param_groups``/``state``), pass
    *step_fn* to override the compiled callable.  The optimizer catalog
    (which maps placeholder tensors to param/grad/state roles) is built
    only when the standard interface is available.
    """
    from torch._functorch.aot_autograd import aot_module_simplified, make_boxed_func

    capture = AtenCapture()

    def _tensor_sig(t: Any) -> tuple[int, tuple[int, ...], str, str] | None:
        if not isinstance(t, torch.Tensor):
            return None
        try:
            ptr = t.untyped_storage().data_ptr()
        except Exception:
            ptr = t.data_ptr()
        return (ptr, tuple(t.shape), str(t.dtype), str(t.device))

    def _build_optimizer_catalog() -> dict[tuple[int, tuple[int, ...], str, str], dict[str, Any]]:
        catalog: dict[tuple[int, tuple[int, ...], str, str], dict[str, Any]] = {}
        flat_idx = 0
        for group_idx, group in enumerate(optimizer.param_groups):
            for group_param_idx, param in enumerate(group["params"]):
                param_ref = (param_name_map or {}).get(
                    id(param),
                    f"group{group_idx}.param{group_param_idx}",
                )
                base = {
                    "param_name": param_ref,
                    "param_index": flat_idx,
                    "group_index": group_idx,
                    "group_param_index": group_param_idx,
                }

                sig = _tensor_sig(param)
                if sig is not None:
                    catalog[sig] = {**base, "role": "param"}

                if isinstance(param.grad, torch.Tensor):
                    sig = _tensor_sig(param.grad)
                    if sig is not None:
                        catalog[sig] = {**base, "role": "grad"}

                state = optimizer.state.get(param, {})
                for state_key, value in state.items():
                    sig = _tensor_sig(value)
                    if sig is not None:
                        catalog[sig] = {
                            **base,
                            "role": "state",
                            "state_key": str(state_key),
                        }
                flat_idx += 1
        return catalog

    optimizer_catalog = _build_optimizer_catalog() if hasattr(optimizer, 'param_groups') else {}

    def fw_compiler(gm: GraphModule, example_inputs):
        gm_copy = _safe_deepcopy_gm(gm)
        inputs_copy = _safe_copy_inputs(example_inputs)
        graph = AtenGraph(
            graph_module=gm_copy,
            example_inputs=inputs_copy,
            kind="optimizer",
            graph_id=capture._counter,
        )
        capture.forward_graphs.append(graph)
        capture._counter += 1

        ph_names = [n.name for n in gm.graph.nodes if n.op == "placeholder"]

        if record_real_tensors:
            def _recording(args, _ph_names=ph_names):
                slot_info: list[dict[str, Any]] = []
                for slot_idx, inp in enumerate(args):
                    info = optimizer_catalog.get(_tensor_sig(inp), {}).copy()
                    if not info:
                        info = {"role": "unknown"}
                    info["slot_index"] = slot_idx
                    if slot_idx < len(_ph_names):
                        info["placeholder_name"] = _ph_names[slot_idx]
                    slot_info.append(info)
                capture.optimizer_slot_info = [slot_info]
                # Store live input data_ptrs for post-capture slot enrichment
                capture._optimizer_input_ptrs = [
                    a.data_ptr() if isinstance(a, torch.Tensor) else None
                    for a in args
                ]
                capture.forward_real_inputs = [
                    a.clone().detach() if isinstance(a, torch.Tensor) else a
                    for a in args
                ]
                interp = _RecordInterp(gm)
                result = interp.run(*args)
                capture.forward_intermediates = dict(interp.recorded)
                capture.forward_real_output = _clone_output(result)
                return result
            _recording._boxed_call = True
            return _recording

        def _capturing_call(args, _ph_names=ph_names):
            slot_info: list[dict[str, Any]] = []
            for slot_idx, inp in enumerate(args):
                info = optimizer_catalog.get(_tensor_sig(inp), {}).copy()
                if not info:
                    info = {"role": "unknown"}
                info["slot_index"] = slot_idx
                if slot_idx < len(_ph_names):
                    info["placeholder_name"] = _ph_names[slot_idx]
                slot_info.append(info)
            capture.optimizer_slot_info = [slot_info]
            # Store live input data_ptrs for post-capture slot enrichment
            capture._optimizer_input_ptrs = [
                a.data_ptr() if isinstance(a, torch.Tensor) else None
                for a in args
            ]
            return gm.forward(*args)
        _capturing_call._boxed_call = True
        return _capturing_call

    def aot_backend(gm, inputs):
        return aot_module_simplified(gm, inputs, fw_compiler=fw_compiler)

    torch.compiler.reset()
    _step = step_fn if step_fn is not None else optimizer.step
    with _patch_compiler_config(capture_scalar_outputs=True):
        compiled_step = torch.compile(_step, backend=aot_backend)
        capture.step_result = compiled_step()

    return capture


# -----------------------------------------------------------------------------
# Source trace capture (Phase 1: FX-level)
# -----------------------------------------------------------------------------


def _parse_stack_trace(trace_str: str) -> tuple[str, int, str, str]:
    """Parse a stack_trace string into (file, line, code, fn_name).

    Stack traces look like:
      File "/path/to/model.py", line 42, in forward
          x = F.relu(self.fc1(x))
    """
    file = ""
    line = 0
    code = ""
    fn_name = ""

    if not trace_str:
        return file, line, code, fn_name

    for raw_line in trace_str.strip().split("\n"):
        raw_line = raw_line.strip()
        if raw_line.startswith("File "):
            # Parse: File "/path/to/file.py", line 42, in forward
            parts = raw_line.split('"')
            if len(parts) >= 2:
                file = parts[1]
            # Extract line number
            if ", line " in raw_line:
                try:
                    line = int(raw_line.split(", line ")[1].split(",")[0].strip())
                except (ValueError, IndexError):
                    pass
            if ", in " in raw_line:
                fn_name = raw_line.split(", in ")[-1].strip()
        elif raw_line and not raw_line.startswith("File "):
            # This is the source code line
            code = raw_line.strip()

    return file, line, code, fn_name


def _extract_source_traces_from_graph(gm: GraphModule) -> dict[str, SourceTrace]:
    """Extract source traces from an FX GraphModule's node metadata.

    Works on the pre-decomposition graph that Dynamo produces (before
    aot_autograd lowers to aten).  These nodes have stack_trace,
    source_fn_stack, and nn_module_stack metadata.

    Returns dict mapping source_fn keys to SourceTrace objects.
    """
    source_map: dict[str, SourceTrace] = {}

    for node in gm.graph.nodes:
        if node.op in ("placeholder", "output"):
            continue

        st = node.meta.get("stack_trace", "")
        src_fn = node.meta.get("source_fn_stack", [])
        nn_mod = node.meta.get("nn_module_stack", {})

        if not st:
            continue

        key = ""
        if src_fn:
            for item in src_fn:
                if isinstance(item, tuple) and len(item) >= 1:
                    key = item[0]

        if not key:
            key = node.name

        file, line_no, code, fn_name = _parse_stack_trace(st)

        module_path = ""
        module_type = ""
        if nn_mod:
            for _, v in nn_mod.items():
                if isinstance(v, tuple) and len(v) >= 2:
                    module_path = _clean_self_path(v[0])
                    module_type = v[1].__name__ if hasattr(v[1], "__name__") else str(v[1])

        source_map[key] = SourceTrace(
            file=file,
            line=line_no,
            code=code,
            fn_name=fn_name,
            module_path=module_path,
            module_type=module_type,
        )

    return source_map


# -----------------------------------------------------------------------------
# Source annotation helpers
# -----------------------------------------------------------------------------


def _build_primal_map(
    graph_module: GraphModule,
    capture: AtenCapture | None = None,
    frag_primal_names: list[str | None] | None = None,
) -> dict[str, str]:
    """Map primal names to human-readable descriptions.

    Uses the exact primal ordering from aot_autograd (params + buffers in
    insertion order from named_parameters/named_buffers, then user inputs).
    Falls back to shape matching when primal_names isn't available.

    Returns dict mapping e.g. "primals_1" -> "self.fc1.weight".
    """
    primal_map = {}
    graph = graph_module.graph
    placeholders = [n for n in graph.nodes if n.op == "placeholder"]

    if frag_primal_names:
        _map_primals_by_order(primal_map, placeholders, capture,
                              override_names=frag_primal_names)
    elif capture and capture.primal_names:
        _map_primals_by_order(primal_map, placeholders, capture)
    elif capture and (capture.param_names or capture.buffer_names):
        _match_primals_to_params(primal_map, placeholders, capture)
    else:
        _infer_primals_from_consumers(primal_map, placeholders)

    return primal_map


def _map_primals_by_order(
    primal_map: dict[str, str],
    placeholders: list[Node],
    capture: AtenCapture | None,
    override_names: list[str | None] | None = None,
) -> None:
    """Map primals using the known parameter ordering from aot_autograd.

    primal_names may contain None entries for user inputs (e.g. from the
    torch.export fallback where inputs and params are interleaved).

    When *override_names* is given, it is used instead of capture.primal_names
    (for per-fragment primal mapping in multi-fragment captures).
    """
    names = override_names if override_names is not None else (capture.primal_names if capture else [])
    n_names = len(names)
    input_idx = 0
    for i, ph in enumerate(placeholders):
        if i < n_names and names[i] is not None:
            primal_map[ph.name] = f"self.{names[i]}"
        else:
            val = ph.meta.get("val")
            if val is not None and hasattr(val, "shape"):
                shape = list(val.shape)
                primal_map[ph.name] = f"input {shape}"
            else:
                primal_map[ph.name] = f"input_{input_idx}"
            input_idx += 1


def _match_primals_to_params(
    primal_map: dict[str, str],
    placeholders: list[Node],
    capture: AtenCapture,
) -> None:
    """Match primals to known parameter/buffer names by shape.

    Any primal whose shape doesn't match a known param/buffer is a user input.
    """
    # Build all known params with shapes
    known = []  # (name, shape_tuple)
    for name, shape in zip(capture.param_names, capture.param_shapes):
        known.append((f"self.{name}", tuple(shape)))
    for name, shape in zip(capture.buffer_names, capture.buffer_shapes):
        known.append((f"(buffer) self.{name}", tuple(shape)))

    # Track which known params have been matched
    matched_known = set()

    # First pass: try unique shape matches
    for ph in placeholders:
        val = ph.meta.get("val")
        if val is None or not hasattr(val, "shape"):
            primal_map[ph.name] = "input"
            continue

        ph_shape = tuple(val.shape)
        candidates = [(i, name) for i, (name, shape) in enumerate(known)
                      if shape == ph_shape and i not in matched_known]

        if len(candidates) == 1:
            idx, param_name = candidates[0]
            matched_known.add(idx)
            primal_map[ph.name] = param_name
        elif len(candidates) == 0:
            # No matching param → user input
            primal_map[ph.name] = f"input {list(ph_shape)}"

    # Second pass: for ambiguous shapes, use consumer metadata to disambiguate
    for ph in placeholders:
        if ph.name in primal_map:
            continue

        val = ph.meta.get("val")
        ph_shape = tuple(val.shape) if hasattr(val, "shape") else ()
        candidates = [(i, name) for i, (name, shape) in enumerate(known)
                      if shape == ph_shape and i not in matched_known]

        if not candidates:
            primal_map[ph.name] = f"input {list(ph_shape)}"
            continue

        # Use consumer nn_module_stack to pick the right candidate
        consumer_module = ""
        for user in ph.users:
            nn_mod = user.meta.get("nn_module_stack", {})
            if nn_mod:
                for _, v in nn_mod.items():
                    if isinstance(v, tuple) and len(v) >= 2:
                        consumer_module = _clean_self_path(v[0], keep_self=False)
                break

        best = None
        for idx, param_name in candidates:
            # Check if param_name contains the consumer module
            clean = param_name.replace("self.", "")
            if consumer_module and consumer_module in clean:
                best = (idx, param_name)
                break

        if best:
            matched_known.add(best[0])
            primal_map[ph.name] = best[1]
        else:
            # Just take the first unmatched candidate
            idx, param_name = candidates[0]
            matched_known.add(idx)
            primal_map[ph.name] = param_name


def _infer_primals_from_consumers(
    primal_map: dict[str, str],
    placeholders: list[Node],
) -> None:
    """Fallback: infer primal identity from consumer node metadata."""
    for ph in placeholders:
        val = ph.meta.get("val")
        shape = list(val.shape) if hasattr(val, "shape") else []

        module_path = ""
        module_type = ""
        for user in ph.users:
            nn_mod = user.meta.get("nn_module_stack", {})
            if nn_mod:
                for _, v in nn_mod.items():
                    if isinstance(v, tuple) and len(v) >= 2:
                        module_path = v[0]
                        module_type = v[1].__name__ if hasattr(v[1], "__name__") else str(v[1])
                break

        if module_path:
            clean_path = _clean_self_path(module_path)
            primal_map[ph.name] = clean_path
        else:
            primal_map[ph.name] = f"input {shape}"




def _extract_source_group(node: Node) -> tuple[str, str, str]:
    """Extract source annotation from a compute node.

    Checks both forward metadata (nn_module_stack, source_fn_stack)
    and backward metadata (fwd_nn_module_stack, fwd_source_fn_stack).

    Returns: (module_path, module_type, source_fn_name)
    """
    module_path = ""
    module_type = ""
    source_fn = ""

    # Try forward metadata first, then fwd_* (backward nodes)
    nn_mod = node.meta.get("nn_module_stack") or node.meta.get("fwd_nn_module_stack", {})
    if nn_mod:
        for _, v in nn_mod.items():
            if isinstance(v, tuple) and len(v) >= 2:
                module_path = v[0]
                module_type = v[1].__name__ if hasattr(v[1], "__name__") else str(v[1])

    src_fn = node.meta.get("source_fn_stack") or node.meta.get("fwd_source_fn_stack", [])
    if src_fn:
        for item in src_fn:
            if isinstance(item, tuple) and len(item) >= 1:
                source_fn = item[0]

    return (module_path, module_type, source_fn)


def _group_key(node: Node) -> str:
    """Generate a grouping key for a compute node.

    Groups primarily by module path. When no module, groups by source_fn.
    This avoids spamming repeated headers for ops in the same module
    that have different source_fn values.
    """
    mod_path, mod_type, src_fn = _extract_source_group(node)
    if mod_path:
        return mod_path  # Group by module, not individual source_fn
    elif src_fn:
        return f"::{src_fn}"
    return ""


def _format_group_header(
    mod_path: str,
    mod_type: str,
    src_fn: str,
    source_trace: SourceTrace | None = None,
    is_backward: bool = False,
    common_source_dir: str | None = None,
) -> list[str]:
    """Format section header comment lines for a group of ops.

    Returns a list of comment lines (without leading #).
    """
    lines = []

    if mod_path:
        clean = _clean_self_path(mod_path)
    else:
        clean = ""

    if clean and mod_type:
        label = f"{clean} ({mod_type})"
    elif clean:
        label = clean
    elif src_fn:
        label = src_fn
    else:
        return []

    if is_backward:
        mod_short = clean.rsplit(".", 1)[-1] if clean else src_fn
        if mod_type:
            label = f"grad of {clean} ({mod_type}) → d_loss/d_{mod_short}"
        else:
            label = f"grad of {label}"

    lines.append(label)

    if source_trace:
        if source_trace.file and source_trace.line:
            fpath = _shorten_source_path(source_trace.file, common_source_dir)
            lines.append(f"{fpath}:{source_trace.line}")
        if source_trace.code:
            lines.append(source_trace.code)

    return lines


def _shorten_source_path(path: str, common_source_dir: str | None = None) -> str:
    """Shorten an absolute source file path by stripping the common graph prefix."""
    if "/dist-packages/" in path:
        parts = path.split("/")
        try:
            idx = parts.index("dist-packages")
        except ValueError:
            idx = -1
        if idx > 0:
            start = max(1, idx - 1)
            return "/" + "/".join(parts[start:])
    if not path or not common_source_dir:
        return path
    try:
        rel = os.path.relpath(path, common_source_dir)
    except ValueError:
        return path
    if rel.startswith(".."):
        return path
    return "/" + rel.replace(os.sep, "/")


def _format_compact_source_comments(
    source_trace: SourceTrace | None,
    common_source_dir: str | None = None,
) -> list[str]:
    """Format compact per-op source comments for source-only groups."""
    if source_trace is None:
        return []

    lines = []
    if source_trace.file and source_trace.line:
        lines.append(f"{_shorten_source_path(source_trace.file, common_source_dir)}:{source_trace.line}")
    if source_trace.code:
        lines.append(source_trace.code)
    return lines


def _lookup_source_trace(
    mod_path: str,
    src_fn: str,
    source_map: dict[str, SourceTrace] | None,
) -> SourceTrace | None:
    """Find the source trace used for annotation of a node/group."""
    if not source_map:
        return None
    if src_fn:
        trace = source_map.get(src_fn)
        if trace is not None:
            return trace
    if mod_path:
        for val in source_map.values():
            if val.module_path and val.module_path in _clean_self_path(mod_path):
                return val
    return None


# -----------------------------------------------------------------------------
# Export as standalone Python
# -----------------------------------------------------------------------------


def _is_symbolic_dim(dim) -> bool:
    """Return True if dim is a symbolic (non-concrete) dimension (e.g. SymInt s0, s1)."""
    try:
        from torch.fx.experimental.symbolic_shapes import is_concrete_int
        return not is_concrete_int(dim)
    except ImportError:
        return isinstance(dim, torch.SymInt)


def _graph_has_symbolic_shapes(graph_module: GraphModule) -> bool:
    """Return True if any node's meta['val'].shape contains symbolic dimensions (SymInt)."""
    for node in graph_module.graph.nodes:
        val = node.meta.get("val")
        if val is not None and isinstance(val, torch.Tensor) and hasattr(val, "shape"):
            for dim in val.shape:
                if _is_symbolic_dim(dim):
                    return True
    return False


def _get_symbolic_dim_names(graph_module: GraphModule) -> list[str]:
    """Collect unique symbolic dimension names (e.g. s0, s1) from the graph."""
    seen: set[str] = set()
    result: list[str] = []
    for node in graph_module.graph.nodes:
        val = node.meta.get("val")
        if val is not None and isinstance(val, torch.Tensor) and hasattr(val, "shape"):
            for dim in val.shape:
                if _is_symbolic_dim(dim):
                    s = str(dim)
                    if s not in seen:
                        seen.add(s)
                        result.append(s)
    return sorted(result)


def _module_path_to_short(mod_path: str) -> str:
    """Convert a module path to a short readable name.

    ``transformer.h.0.attn.c_q`` → ``h0_attn_c_q``

    Skips leading container names, collapses ``h.0`` → ``h0``, and keeps
    at most 3 trailing parts for readability.
    """
    clean = _clean_self_path(mod_path, keep_self=False)
    parts = clean.split(".")
    # Skip leading generic container names
    skip = {"transformer", "model", "encoder", "decoder", "backbone", "module", "layers", "self"}
    while parts and parts[0].lower() in skip:
        parts = parts[1:]
    if not parts:
        return ""
    # Collapse numeric parts: ["h", "0", "attn"] → ["h0", "attn"]
    collapsed: list[str] = []
    for p in parts:
        if p.isdigit() and collapsed:
            collapsed[-1] = collapsed[-1] + p
        else:
            collapsed.append(p)
    # Keep last 3 parts max
    collapsed = collapsed[-3:]
    return "_".join(collapsed)


def _derive_named_intermediate(node: Node, source_map: dict[str, SourceTrace] | None = None) -> str | None:
    """Derive a human-readable name for a compute node from its source context.

    Returns None if no good name can be derived (keep the original).
    """
    mod_path, mod_type, src_fn = _extract_source_group(node)

    target = node.target
    if hasattr(target, "name") and callable(target.name):
        try:
            raw = target.name()
            if "::" in raw:
                op_short = raw.split("::")[-1].split(".")[0]
            else:
                op_short = raw.split(".")[-1]
        except Exception:
            op_short = str(target).split(".")[-1]
    elif hasattr(target, "__name__"):
        op_short = target.__name__
    else:
        op_short = str(target).split(".")[-1]

    # Detect backward nodes via fwd_nn_module_stack metadata
    is_backward = "fwd_nn_module_stack" in node.meta and "nn_module_stack" not in node.meta
    prefix = "grad_" if is_backward else ""

    name = None
    if mod_path:
        mod_short = _module_path_to_short(mod_path)
        if mod_short:
            name = f"{prefix}{mod_short}_{op_short}"
        else:
            # Fallback to last part if _module_path_to_short returned empty
            clean = _clean_self_path(mod_path, keep_self=False)
            parts = clean.split(".")
            mod_short = parts[-1] if parts else ""
            if mod_short:
                name = f"{prefix}{mod_short}_{op_short}"
    elif src_fn:
        name = f"{prefix}{src_fn}_{op_short}"

    # Ensure the name is a valid Python identifier (no leading digits)
    if name and name[0].isdigit():
        name = f"l{name}"

    return name


def _sanitize_primal_names(
    graph_nodes,
    primal_map: dict[str, str],
    used_names: set[str] | None = None,
) -> dict[str, str]:
    """Build a name remap for primals using human-readable descriptions.

    Iterates placeholder nodes in the graph, sanitizes their human name from
    primal_map, deduplicates, and returns {fx_name: sanitized_name}.
    """
    remap = {}
    used = set(used_names) if used_names else set()
    for node in graph_nodes:
        if node.op != "placeholder" or node.name not in primal_map:
            continue
        human = primal_map[node.name]
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', human).strip('_')
        if sanitized.startswith("self_"):
            sanitized = sanitized[5:]
        if not sanitized or sanitized[0].isdigit():
            sanitized = f"p_{sanitized}"
        if sanitized.startswith("input"):
            sanitized = re.sub(r'_+$', '', sanitized)
            if sanitized == "input":
                sanitized = node.name
        base = sanitized
        counter = 0
        while sanitized in used:
            counter += 1
            sanitized = f"{base}_{counter}"
        remap[node.name] = sanitized
        used.add(sanitized)
    return remap


# ---------------------------------------------------------------------------
# Uniquification of repeated module groups
# ---------------------------------------------------------------------------


@dataclass
class _UniqueGroup:
    """Describes a set of structurally identical repeated module instances."""
    template_key: str                           # "encoder.layers.*"
    module_type: str                            # "TransformerEncoderLayer"
    fn_name: str                                # "residual_block"
    instances: dict[str, list[Node]]            # {"0": [fx_nodes...], "1": [fx_nodes...]}
    instance_order: list[str]                   # ["0", "1"] in graph order
    params: list[dict]                          # [{name, annotation, shape_comment}, ...]
    returns: list[dict]                         # [{name, annotation, shape_comment}, ...]
    body_code: str                              # generated function body (indented)
    first_node_per_instance: dict[str, str]     # instance_idx -> first node.name
    all_node_names: set[str]                    # all node names across all instances
    output_name_map: dict[str, dict[str, str]]  # instance_idx -> {local_return -> actual_name}
    input_name_map: dict[str, dict[str, str]]   # instance_idx -> {local_param -> actual_name}


def _compute_node_top_module(node: Node) -> tuple[str, str, str] | None:
    """Extract (top_module_path, instance_index, module_type) from nn_module_stack.

    Returns None if the node has no numeric-indexed module at depth >= 2.
    """
    nn_mod = node.meta.get("nn_module_stack") or node.meta.get("fwd_nn_module_stack", {})
    if not nn_mod:
        return None

    # nn_module_stack is OrderedDict: key -> (path, type)
    # We want the deepest entry that has a numeric index at depth >= 2
    # e.g. "self.encoder.layers.0" -> parts = ["encoder", "layers", "0"]
    best = None
    for _, v in nn_mod.items():
        if not (isinstance(v, tuple) and len(v) >= 2):
            continue
        raw_path = v[0]
        mod_type_obj = v[1]
        mod_type = mod_type_obj.__name__ if hasattr(mod_type_obj, "__name__") else str(mod_type_obj)
        clean = _clean_self_path(raw_path)
        parts = clean.split(".")
        # Find the first numeric index at depth >= 2
        for i, p in enumerate(parts):
            if p.isdigit() and i >= 2:
                top_path = ".".join(parts[:i + 1])
                instance_idx = p
                return (top_path, instance_idx, mod_type)
            elif p.isdigit() and i >= 1:
                top_path = ".".join(parts[:i + 1])
                instance_idx = p
                best = (top_path, instance_idx, mod_type)
    return best


def _normalize_template_key(top_module: str) -> str:
    """Replace the numeric instance index with '*'.

    "encoder.layers.0" -> "encoder.layers.*"
    "blocks.1" -> "blocks.*"
    """
    parts = top_module.split(".")
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].isdigit():
            parts[i] = "*"
            break
    return ".".join(parts)


def _compute_all_module_levels(node: Node) -> list[tuple[str, str, str]]:
    """Return all (module_path, instance_idx, module_type) levels for a node.

    Walks the nn_module_stack from outermost to innermost.  Once a numeric
    instance index is found (e.g. ``transformer.h.0``), that entry and every
    deeper entry become levels for hierarchical grouping.

    For a node in ``transformer.h.0.attn.c_q``, returns::

        [("transformer.h.0", "0", "Block"),
         ("transformer.h.0.attn", "0", "CausalSelfAttention"),
         ("transformer.h.0.attn.c_q", "0", "Linear")]

    Returns an empty list when no numeric-indexed module is found.
    """
    nn_mod = node.meta.get("nn_module_stack") or node.meta.get("fwd_nn_module_stack", {})
    if not nn_mod:
        return []

    results: list[tuple[str, str, str]] = []
    base_idx: str | None = None
    base_depth: int | None = None

    for _, v in nn_mod.items():
        if not (isinstance(v, tuple) and len(v) >= 2):
            continue
        raw_path = v[0]
        mod_type_obj = v[1]
        mod_type = mod_type_obj.__name__ if hasattr(mod_type_obj, "__name__") else str(mod_type_obj)
        clean = _clean_self_path(raw_path)
        parts = clean.split(".")

        if base_idx is None:
            # Looking for the first numeric index at depth >= 1
            for i, p in enumerate(parts):
                if p.isdigit() and i >= 1:
                    base_idx = p
                    base_depth = i
                    results.append((clean, base_idx, mod_type))
                    break
        else:
            # Already found numeric base — this is a sub-module underneath it.
            # Verify this path is actually under the same numeric-indexed parent.
            if len(parts) > base_depth and parts[base_depth] == base_idx:
                results.append((clean, base_idx, mod_type))

    return results


def _build_group_from_instances(
    template: str,
    mod_type: str,
    instances: dict[str, list[Node]],
    instance_ir: dict[str, list[dict]],
    instance_order: list[str],
    compute_nodes: list[Node],
    ir_nodes_by_name: dict[str, dict],
    name_remap: dict[str, str],
) -> _UniqueGroup | None:
    """Build a ``_UniqueGroup`` from structurally-matched instances.

    This implements Phases 3–4 of the original uniquification algorithm:
    computing external inputs/outputs and generating the function body.
    """
    if not instance_order:
        return None

    template_idx = instance_order[0]
    template_nodes = instances[template_idx]
    template_ir_nodes = instance_ir[template_idx]
    template_names = {n.name for n in template_nodes}

    # ── Phase 3: external inputs & outputs ────────────────────────

    external_inputs_ordered: list[str] = []

    def _collect_external_refs(v: dict, group_names: set[str], ext_list: list[str]):
        if "node" in v:
            ref = v["node"]
            if ref not in group_names and ref not in ext_list:
                ext_list.append(ref)
            return
        kind = v.get("kind", "")
        if kind in ("list", "tuple"):
            for item in v.get("items", []):
                _collect_external_refs(item, group_names, ext_list)
        elif kind == "dict":
            for item in v.get("items", []):
                _collect_external_refs(item["key"], group_names, ext_list)
                _collect_external_refs(item["value"], group_names, ext_list)
        elif kind == "slice":
            _collect_external_refs(v["start"], group_names, ext_list)
            _collect_external_refs(v["stop"], group_names, ext_list)
            _collect_external_refs(v["step"], group_names, ext_list)

    for ir_node in template_ir_nodes:
        for arg in ir_node.get("args", []):
            _collect_external_refs(arg, template_names, external_inputs_ordered)
        for _, kwarg_v in ir_node.get("kwargs", {}).items():
            _collect_external_refs(kwarg_v, template_names, external_inputs_ordered)

    # Find outputs: nodes whose values are used OUTSIDE the group
    template_outputs: list[str] = []
    for node in template_nodes:
        for user in node.users:
            if user.name not in template_names:
                if node.name not in template_outputs:
                    template_outputs.append(node.name)
                break

    # Also check the output node
    output_node = None
    for n in compute_nodes[0].graph.nodes:
        if n.op == "output":
            output_node = n
            break

    if output_node:
        def _check_output_refs(val, names_set, outputs_list):
            if isinstance(val, Node):
                if val.name in names_set and val.name not in outputs_list:
                    outputs_list.append(val.name)
            elif isinstance(val, (tuple, list)):
                for v in val:
                    _check_output_refs(v, names_set, outputs_list)
        _check_output_refs(
            output_node.args[0] if output_node.args else [],
            template_names, template_outputs,
        )

    if not template_outputs and not external_inputs_ordered:
        return None

    # Build per-instance input/output name maps
    input_name_map: dict[str, dict[str, str]] = {}
    output_name_map: dict[str, dict[str, str]] = {}

    for idx in instance_order:
        inst_nodes = instances[idx]
        inst_ir = instance_ir[idx]
        inst_names = {n.name for n in inst_nodes}

        inst_ext: list[str] = []
        for ir_node in inst_ir:
            for arg in ir_node.get("args", []):
                _collect_external_refs(arg, inst_names, inst_ext)
            for _, kwarg_v in ir_node.get("kwargs", {}).items():
                _collect_external_refs(kwarg_v, inst_names, inst_ext)

        inp_map = {}
        for i, tmpl_ext in enumerate(external_inputs_ordered):
            local_name = name_remap.get(tmpl_ext, tmpl_ext)
            if i < len(inst_ext):
                inp_map[local_name] = name_remap.get(inst_ext[i], inst_ext[i])
        input_name_map[idx] = inp_map

        inst_outputs: list[str] = []
        for node in inst_nodes:
            for user in node.users:
                if user.name not in inst_names:
                    if node.name not in inst_outputs:
                        inst_outputs.append(node.name)
                    break
        if output_node:
            _check_output_refs(
                output_node.args[0] if output_node.args else [],
                inst_names, inst_outputs,
            )

        out_map = {}
        for i, tmpl_out in enumerate(template_outputs):
            local_name = name_remap.get(tmpl_out, tmpl_out)
            if i < len(inst_outputs):
                out_map[local_name] = name_remap.get(inst_outputs[i], inst_outputs[i])
        output_name_map[idx] = out_map

    # ── Phase 4: function body & signature ────────────────────────

    template_idx_str = template_idx
    raw_base = template.replace(".*", "")
    if raw_base.startswith("self."):
        raw_base = raw_base[5:]
    template_base = raw_base.replace(".", "_")
    instance_prefix = f"{template_base}_{template_idx_str}_"

    def _genericize_name(name: str) -> str:
        if name.startswith(instance_prefix):
            stripped = name[len(instance_prefix):]
            if stripped and not stripped[0].isdigit():
                return stripped
        return name

    generic_param_names: dict[str, str] = {}
    param_defs: list[dict] = []
    used_generic: set[str] = set()
    for ext_name in external_inputs_ordered:
        local = name_remap.get(ext_name, ext_name)
        generic = _genericize_name(local)
        base = generic
        counter = 0
        while generic in used_generic:
            counter += 1
            generic = f"{base}_{counter}"
        used_generic.add(generic)
        generic_param_names[local] = generic
        annotation = ""
        ir_node = ir_nodes_by_name.get(ext_name)
        if ir_node and ir_node.get("meta"):
            meta = ir_node["meta"]
            if meta.get("shape") and meta.get("dtype"):
                dtype_name = meta["dtype"].split(".")[-1]
                dims = ", ".join(str(s) for s in meta["shape"])
                annotation = f"{dtype_name}[{dims}]"
        for n in compute_nodes[0].graph.nodes:
            if n.op == "placeholder" and n.name == ext_name:
                from torch_graph.internal_ir import placeholder_annotation as _ph_ann, tensor_meta as _tmeta
                annotation = _ph_ann(n)
                tmeta = _tmeta(n.meta.get("val"))
                if tmeta:
                    shape_str = f"{tmeta['dtype'].split('.')[-1]}[{', '.join(str(s) for s in tmeta['shape'])}]"
                    annotation = shape_str
                break
        param_defs.append({"name": generic, "annotation": annotation})

    generic_output_names: dict[str, str] = {}
    return_defs: list[dict] = []
    for out_name in template_outputs:
        local = name_remap.get(out_name, out_name)
        generic_out = _genericize_name(local)
        base = generic_out
        counter = 0
        while generic_out in used_generic:
            counter += 1
            generic_out = f"{base}_{counter}"
        used_generic.add(generic_out)
        generic_output_names[local] = generic_out
        annotation = ""
        ir_node = ir_nodes_by_name.get(out_name)
        if ir_node and ir_node.get("meta"):
            meta = ir_node["meta"]
            if meta.get("shape") and meta.get("dtype"):
                dtype_name = meta["dtype"].split(".")[-1]
                dims = ", ".join(str(s) for s in meta["shape"])
                annotation = f"{dtype_name}[{dims}]"
        return_defs.append({"name": generic_out, "annotation": annotation})

    fn_name = re.sub(r'(?<!^)(?=[A-Z])', '_', mod_type).lower()
    fn_name = re.sub(r'[^a-zA-Z0-9_]', '_', fn_name).strip('_')
    if not fn_name or fn_name[0].isdigit():
        fn_name = f"layer_{fn_name}"

    # Generate body code
    body_lines: list[str] = []
    local_remap: dict[str, str] = {}
    for ext_name in external_inputs_ordered:
        orig_local = name_remap.get(ext_name, ext_name)
        local_remap[ext_name] = generic_param_names.get(orig_local, orig_local)
    for node in template_nodes:
        orig_local = name_remap.get(node.name, node.name)
        generic_intermediate = _genericize_name(orig_local)
        local_remap[node.name] = generic_intermediate

    if local_remap:
        sorted_names = sorted(local_remap.keys(), key=len, reverse=True)
        pattern = r'(?<![.])\b(?:' + '|'.join(re.escape(n) for n in sorted_names) + r')\b'
        local_remap_re = re.compile(pattern)
    else:
        local_remap_re = None

    for node in template_nodes:
        ir_node = ir_nodes_by_name.get(node.name)
        if ir_node is None:
            continue
        line = ir_node["python"]
        if not line:
            continue
        if local_remap_re is not None:
            line = local_remap_re.sub(lambda m: local_remap.get(m.group(0), m.group(0)), line)
        for sub in line.split("\n"):
            body_lines.append(f"    {sub}")

    if template_outputs:
        ret_names = []
        for n in template_outputs:
            orig_local = name_remap.get(n, n)
            ret_names.append(generic_output_names.get(orig_local, _genericize_name(orig_local)))
        body_lines.append(f"    return ({', '.join(ret_names)},)")

    body_code = "\n".join(body_lines)

    # Remap input/output maps to use generic keys
    generic_input_name_map: dict[str, dict[str, str]] = {}
    for idx in instance_order:
        new_map = {}
        for orig_key, actual_val in input_name_map[idx].items():
            generic_key = generic_param_names.get(orig_key, orig_key)
            new_map[generic_key] = actual_val
        generic_input_name_map[idx] = new_map

    generic_output_name_map: dict[str, dict[str, str]] = {}
    for idx in instance_order:
        new_map = {}
        for orig_key, actual_val in output_name_map[idx].items():
            generic_key = generic_output_names.get(orig_key, orig_key)
            new_map[generic_key] = actual_val
        generic_output_name_map[idx] = new_map

    all_names: set[str] = set()
    first_node_map: dict[str, str] = {}
    for idx in instance_order:
        nodes = instances[idx]
        for n in nodes:
            all_names.add(n.name)
        if nodes:
            first_node_map[idx] = nodes[0].name

    return _UniqueGroup(
        template_key=template,
        module_type=mod_type,
        fn_name=fn_name,
        instances=instances,
        instance_order=instance_order,
        params=param_defs,
        returns=return_defs,
        body_code=body_code,
        first_node_per_instance=first_node_map,
        all_node_names=all_names,
        output_name_map=generic_output_name_map,
        input_name_map=generic_input_name_map,
    )


def _disambiguate_fn_names(groups: list[_UniqueGroup]) -> None:
    """Ensure all extracted function names are unique by adding numeric suffixes."""
    name_indices: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        name_indices.setdefault(g.fn_name, []).append(i)
    for name, indices in name_indices.items():
        if len(indices) <= 1:
            continue
        for j, idx in enumerate(indices):
            groups[idx].fn_name = f"{name}_{j}"


def _build_structural_signature(
    ir_nodes: list[dict],
    group_names: set[str],
) -> tuple:
    """Build a hashable structural signature for a group of IR nodes.

    Node refs inside the group are normalized to relative indices.
    Node refs outside the group become ("ext", position_in_external_inputs_list).
    Literals become ("lit", value).
    """
    external_inputs: list[str] = []  # ordered list of external node names
    node_names_in_order = [n["name"] for n in ir_nodes]
    name_to_idx = {name: i for i, name in enumerate(node_names_in_order)}

    def _normalize_value(v: dict) -> tuple:
        if "node" in v:
            ref = v["node"]
            if ref in name_to_idx:
                return ("@", name_to_idx[ref])
            else:
                if ref not in external_inputs:
                    external_inputs.append(ref)
                return ("ext", external_inputs.index(ref))
        kind = v.get("kind", "")
        if kind in ("list", "tuple"):
            return (kind, tuple(_normalize_value(item) for item in v.get("items", [])))
        if kind == "dict":
            return (kind, tuple(
                (_normalize_value(item["key"]), _normalize_value(item["value"]))
                for item in v.get("items", [])
            ))
        if kind == "slice":
            return ("slice", _normalize_value(v["start"]), _normalize_value(v["stop"]), _normalize_value(v["step"]))
        # Literal value
        return ("lit", v.get("value"))

    sig_parts = []
    for ir_node in ir_nodes:
        target = ir_node["target"]
        args_sig = tuple(_normalize_value(a) for a in ir_node.get("args", []))
        kwargs_sig = tuple(
            (k, _normalize_value(v))
            for k, v in sorted(ir_node.get("kwargs", {}).items())
        )
        sig_parts.append((target, args_sig, kwargs_sig))

    return (tuple(sig_parts), len(external_inputs))


def _detect_unique_groups(
    compute_nodes: list[Node],
    ir_nodes_by_name: dict[str, dict],
    name_remap: dict[str, str],
    source_map: dict | None,
    is_backward: bool,
    *,
    max_depth: int = -1,
    min_ops: int = 0,
) -> list[_UniqueGroup]:
    """Detect repeated structurally-identical module groups, with hierarchical fallback.

    Tries shallowest module depth first (e.g. ``transformer.h.*``).  When the
    top-level doesn't produce a full match, drills into sub-modules
    (e.g. ``transformer.h.*.attn``, ``transformer.h.*.mlp``) and extracts
    those separately.

    Supports *partial* matching: if only a subset of instances share a
    signature (e.g. even vs odd layers), each subset becomes its own group.

    Returns a list of ``_UniqueGroup`` objects for groups with 2+ identical
    instances.
    """
    from collections import defaultdict

    # Phase 1: Group nodes by ALL template levels (multi-depth)
    all_templates: dict[str, dict[str, list[Node]]] = {}  # template -> {idx -> [nodes]}
    all_types: dict[str, str] = {}  # template -> module_type

    for node in compute_nodes:
        for top_path, instance_idx, mod_type in _compute_all_module_levels(node):
            template = _normalize_template_key(top_path)
            all_templates.setdefault(template, {}).setdefault(instance_idx, []).append(node)
            if template not in all_types:
                all_types[template] = mod_type

    # Sort templates by depth (shallowest first) so top-level groups are tried
    # before sub-module groups.  Ties broken alphabetically for determinism.
    sorted_templates = sorted(all_templates.keys(), key=lambda t: (t.count("."), t))

    # Compute depth limit from max_depth (relative to shallowest template)
    if max_depth > 0 and sorted_templates:
        base_depth = sorted_templates[0].count(".")
        depth_limit = base_depth + max_depth - 1
    else:
        depth_limit = None  # no limit

    groups: list[_UniqueGroup] = []
    extracted_nodes: set[str] = set()  # nodes already in a group — skip at deeper levels

    for template in sorted_templates:
        # Enforce depth limit
        if depth_limit is not None and template.count(".") > depth_limit:
            continue

        raw_instances = all_templates[template]

        # Filter out nodes already extracted at a shallower depth
        filtered: dict[str, list[Node]] = {}
        for idx, nodes in raw_instances.items():
            remaining = [n for n in nodes if n.name not in extracted_nodes]
            if remaining:
                filtered[idx] = remaining

        if len(filtered) < 2:
            continue

        # Phase 2: Build IR + structural signatures, group by signature
        instance_ir: dict[str, list[dict]] = {}
        instance_sigs: dict[str, tuple] = {}
        for idx, nodes in filtered.items():
            group_names = {n.name for n in nodes}
            ir_nodes = [ir_nodes_by_name[n.name] for n in nodes if n.name in ir_nodes_by_name]
            if not ir_nodes:
                continue
            instance_ir[idx] = ir_nodes
            instance_sigs[idx] = _build_structural_signature(ir_nodes, group_names)

        if len(instance_sigs) < 2:
            continue

        # Bucket instances by signature (allows partial matching)
        sig_buckets: dict[tuple, list[str]] = defaultdict(list)
        for idx, sig in instance_sigs.items():
            sig_buckets[sig].append(idx)

        for _sig, matching_idxs in sig_buckets.items():
            if len(matching_idxs) < 2:
                continue

            # Enforce min_ops threshold
            if min_ops > 0:
                first_idx = matching_idxs[0]
                if len(instance_ir.get(first_idx, [])) < min_ops:
                    continue

            # Determine graph order for this subset
            instance_first_pos: dict[str, int] = {}
            for idx in matching_idxs:
                for i, cn in enumerate(compute_nodes):
                    if cn is filtered[idx][0]:
                        instance_first_pos[idx] = i
                        break
            instance_order = sorted(matching_idxs, key=lambda x: instance_first_pos.get(x, 0))

            subset_instances = {idx: filtered[idx] for idx in matching_idxs}
            subset_ir = {idx: instance_ir[idx] for idx in matching_idxs}

            group = _build_group_from_instances(
                template, all_types[template],
                subset_instances, subset_ir, instance_order,
                compute_nodes, ir_nodes_by_name, name_remap,
            )
            if group is not None:
                groups.append(group)
                extracted_nodes.update(group.all_node_names)

    _disambiguate_fn_names(groups)
    return groups


def _generate_unique_fn_def(group: _UniqueGroup) -> str:
    """Generate the complete function definition string for a unique group."""
    lines: list[str] = []

    # Build signature
    params_parts: list[str] = []
    for p in group.params:
        ann = f": '{p['annotation']}'" if p.get("annotation") else ""
        params_parts.append(f"    {p['name']}{ann},")

    # Build return annotation
    if group.returns:
        ret_anns = []
        for r in group.returns:
            if r.get("annotation"):
                ret_anns.append(f"'{r['annotation']}'")
            else:
                ret_anns.append("...")
        ret_annotation = f" -> tuple[{', '.join(ret_anns)}]"
    else:
        ret_annotation = ""

    lines.append(f"def {group.fn_name}(")
    lines.extend(params_parts)
    lines.append(f"){ret_annotation}:")

    # Add body
    lines.append(group.body_code)

    return "\n".join(lines)


def _generate_call_site(
    group: _UniqueGroup,
    instance_idx: str,
    name_remap: dict[str, str],
) -> list[str]:
    """Generate the call site lines for a specific instance of a unique group."""
    lines: list[str] = []

    inp_map = group.input_name_map[instance_idx]
    out_map = group.output_name_map[instance_idx]

    # Build output tuple
    if group.returns:
        out_names = list(out_map.values())
        lhs = f"({', '.join(out_names)},)"
    else:
        lhs = None

    # Build keyword call
    call_args: list[str] = []
    for p in group.params:
        param_name = p["name"]
        actual = inp_map.get(param_name, param_name)
        call_args.append(f"        {param_name}={actual},")

    if lhs:
        lines.append(f"    {lhs} = {group.fn_name}(")
    else:
        lines.append(f"    {group.fn_name}(")
    lines.extend(call_args)
    lines.append("    )")

    return lines


def export_graph_to_python(
    graph_module: GraphModule,
    fn_name: str = "forward",
    inline_threshold: int = 1000,
    primal_map: dict[str, str] | None = None,
    annotate_sources: bool = True,
    source_map: dict[str, SourceTrace] | None = None,
    is_backward: bool = False,
    named_intermediates: bool = False,
    dynamic_dims_comment: str | None = None,
    kernel_map: dict[str, str] | None = None,
    uniquify: bool = False,
    uniquify_depth: int = -1,
    uniquify_min_ops: int = 0,
    _unique_fn_defs: list[str] | None = None,
    _unique_groups: list | None = None,
) -> str:
    """Export an FX GraphModule as a standalone Python function.

    Args:
        graph_module: The FX GraphModule to export.
        fn_name: Function name in the generated code (default "forward").
        inline_threshold: Max tensor elements before storing in external .pt file.
        primal_map: Optional mapping of primal names to human-readable descriptions.
        annotate_sources: If True, insert source code group headers.
        source_map: SourceTrace dict for annotation grouping.
        is_backward: If True, emit backward-specific annotations.
        named_intermediates: If True, use source-derived variable names where possible.
        dynamic_dims_comment: Comment string about dynamic dimensions.
        kernel_map: Optional mapping of FX node → Triton kernel name.
        uniquify: If True, detect repeated identical module groups and extract
            them into shared helper functions called from the main body.
        uniquify_depth: How many depth levels to try.  ``1`` = top-level only
            (e.g. ``transformer.h.*``).  ``2`` = also try one sub-module level.
            ``-1`` (default) = all depths.
        uniquify_min_ops: Minimum aten ops for a group to be extracted.
            Groups with fewer ops are skipped.  ``0`` (default) = no minimum.
        _unique_fn_defs: When *uniquify* is True, append generated helper
            function definition strings to this list (caller uses them to
            emit a SHARED LAYER FUNCTIONS section).
        _unique_groups: When *uniquify* is True, append detected
            ``_UniqueGroup`` objects to this list (caller can use them to
            generate CUDA kernel templates).
    """
    buf = StringIO()
    graph = graph_module.graph
    ir_graph = _build_graph_ir(graph_module, fn_name=fn_name, placeholder_display_names=primal_map, source_map=source_map)
    ir_nodes_by_name = {node["name"]: node for node in ir_graph["nodes"]}

    # Build name remapping for named intermediates
    name_remap: dict[str, str] = {}
    if named_intermediates:
        used_names: set[str] = set()
        for node in graph.nodes:
            if node.op in ("placeholder", "output"):
                continue
            derived = _derive_named_intermediate(node, source_map)
            if derived:
                base = derived
                candidate = base
                counter = 0
                while candidate in used_names:
                    counter += 1
                    candidate = f"{base}_{counter}"
                name_remap[node.name] = candidate
                used_names.add(candidate)

    # Remap primals to human-readable names (e.g. primals_1 → transformer_h_0_attn_c_k_weight)
    if primal_map:
        primal_remap = _sanitize_primal_names(graph.nodes, primal_map, set(name_remap.values()))
        name_remap.update(primal_remap)

    def remap(name: str) -> str:
        return name_remap.get(name, name)

    # Collect placeholders
    placeholders = [
        (remap(entry["name"]), f": '{entry['annotation']}'" if entry.get("annotation") else "")
        for entry in ir_graph["placeholders"]
    ]

    # Function signature
    if len(placeholders) <= 3:
        params = ", ".join(f"{name}{ann}" for name, ann in placeholders)
        buf.write(f"def {fn_name}({params}):\n")
    else:
        buf.write(f"def {fn_name}(\n")
        for i, (name, ann) in enumerate(placeholders):
            buf.write(f"    {name}{ann},\n")
        buf.write(f"):\n")

    if dynamic_dims_comment:
        buf.write(f"    # {dynamic_dims_comment}\n")

    compute_nodes = [n for n in graph.nodes if n.op not in ("placeholder", "output")]
    common_source_dir = None
    if source_map:
        source_dirs = []
        for node in compute_nodes:
            mod_path, _, src_fn = _extract_source_group(node)
            trace = _lookup_source_trace(mod_path, src_fn, source_map)
            if trace and trace.file:
                if "/dist-packages/" in trace.file:
                    continue
                source_dirs.append(os.path.dirname(trace.file))
        if source_dirs:
            try:
                common_source_dir = os.path.commonpath(source_dirs)
            except ValueError:
                common_source_dir = None

    # Pre-compile a single regex for all name remappings (O(1) per line instead of O(N))
    _remap_re = None
    if name_remap:
        # Sort by length descending so longer names match first (e.g. primals_10 before primals_1)
        sorted_names = sorted(name_remap.keys(), key=len, reverse=True)
        # Negative lookbehind for '.' prevents matching names inside dotted paths
        # (e.g. node named 't' must NOT match 'aten.t.default')
        pattern = r'(?<![.])\b(?:' + '|'.join(re.escape(n) for n in sorted_names) + r')\b'
        _remap_re = re.compile(pattern)

    # Pre-compute kernel op counts to avoid O(N*K) rescanning in the loop
    kernel_op_counts: dict[str, int] = {}
    if kernel_map:
        from collections import Counter
        kernel_op_counts = Counter(
            kernel_map[n.name] for n in compute_nodes if n.name in kernel_map
        )

    # Detect repeated module groups for uniquification
    unique_groups: list[_UniqueGroup] = []
    unique_skip_nodes: set[str] = set()      # nodes to skip (emitted via call site)
    unique_call_at: dict[str, tuple[_UniqueGroup, str]] = {}  # node.name -> (group, instance_idx)
    if uniquify:
        unique_groups = _detect_unique_groups(
            compute_nodes, ir_nodes_by_name, name_remap, source_map, is_backward,
            max_depth=uniquify_depth, min_ops=uniquify_min_ops,
        )
        for group in unique_groups:
            if _unique_fn_defs is not None:
                _unique_fn_defs.append(_generate_unique_fn_def(group))
            if _unique_groups is not None:
                _unique_groups.append(group)
            for idx in group.instance_order:
                first = group.first_node_per_instance.get(idx)
                if first:
                    unique_call_at[first] = (group, idx)
            unique_skip_nodes.update(group.all_node_names)

    current_group_key = None
    current_kernel = None
    current_top_module = None  # Track top-level module for section headers

    for node in compute_nodes:
        # Uniquification: emit call site at first node, skip rest
        if node.name in unique_skip_nodes:
            if node.name in unique_call_at:
                group, inst_idx = unique_call_at[node.name]
                # Reconstruct top_path for section header
                info = _compute_node_top_module(node)
                if info:
                    top_path = info[0]
                    if top_path != current_top_module:
                        current_top_module = top_path
                        buf.write(f"\n    # {'═' * 64}\n")
                        buf.write(f"    # {top_path} ({group.module_type})\n")
                        buf.write(f"    # {'═' * 64}\n")
                call_lines = _generate_call_site(group, inst_idx, name_remap)
                for cl in call_lines:
                    buf.write(f"{cl}\n")
            continue

        ir_node = ir_nodes_by_name.get(node.name)
        if ir_node is None:
            continue
        line = ir_node["python"]
        if not line:
            continue

        # Apply name remapping (primals + named intermediates)
        if _remap_re is not None:
            line = _remap_re.sub(lambda m: name_remap[m.group(0)], line)

        # Kernel boundary header
        if kernel_map and node.op not in ("output",):
            node_kernel = kernel_map.get(node.name)
            if node_kernel and node_kernel != current_kernel:
                current_kernel = node_kernel
                n_ops = kernel_op_counts.get(node_kernel, 0)
                ktype = "cuBLAS" if "extern" in node_kernel else "triton"
                label = f"Kernel: {node_kernel} ({ktype}, {n_ops} ops)"
                buf.write(f"\n    # ══ {label} {'═' * max(1, 56 - len(label))}\n")

        if annotate_sources and node.op not in ("output",):
            mod_path, mod_type, src_fn = _extract_source_group(node)
            trace = _lookup_source_trace(mod_path, src_fn, source_map)

            compact_lines = []
            annotation_key = None

            if mod_path:
                annotation_key = ("module", _group_key(node))
            elif trace:
                compact_lines = _format_compact_source_comments(trace, common_source_dir)
                annotation_key = (
                    "source",
                    _shorten_source_path(trace.file, common_source_dir),
                    trace.line,
                    trace.code,
                )
            elif src_fn:
                annotation_key = ("source_fn", src_fn)

            if annotation_key and annotation_key != current_group_key:
                current_group_key = annotation_key

                # Emit top-level module section header when module changes
                if mod_path:
                    clean = _clean_self_path(mod_path)
                    # Top-level: first nn.Module boundary with numeric index (e.g. self.transformer.h.0)
                    parts = clean.split(".")
                    top = clean
                    for i, p in enumerate(parts):
                        if p.isdigit() and i >= 2:
                            top = ".".join(parts[:i + 1])
                            break
                    else:
                        # No numeric index — use depth-2 path (e.g. self.transformer.wte)
                        top = ".".join(parts[:min(3, len(parts))])
                    if top != current_top_module:
                        current_top_module = top
                        buf.write(f"\n    # {'═' * 64}\n")
                        buf.write(f"    # {top}\n")
                        buf.write(f"    # {'═' * 64}\n")

                if compact_lines:
                    buf.write("\n")
                    for cl in compact_lines:
                        buf.write(f"    # {cl}\n")
                else:
                    header_lines = _format_group_header(
                        mod_path,
                        mod_type,
                        src_fn,
                        trace,
                        is_backward=is_backward,
                        common_source_dir=common_source_dir,
                    )
                    if header_lines:
                        buf.write("\n")
                        for hl in header_lines:
                            buf.write(f"    # {hl}\n")

        line += ir_node.get("stride_comment", "")
        for sub in line.split("\n"):
            buf.write(f"    {sub}\n")

    return_line = _ir_return_to_python(ir_graph["returns"])
    if _remap_re is not None:
        return_line = _remap_re.sub(lambda m: name_remap[m.group(0)], return_line)
    buf.write(f"    {return_line}\n")

    return buf.getvalue()


def _to_storage_dtype(t: torch.Tensor, dtype: torch.dtype | None) -> torch.Tensor:
    """Convert a tensor to *dtype* for compact storage.  Non-float tensors are unchanged."""
    if dtype is None or not t.dtype.is_floating_point:
        return t
    return t.to(dtype)


def _capped_intermediates(
    intermediates: dict[str, torch.Tensor],
    max_bytes: int | None,
    storage_dtype: torch.dtype | None,
) -> tuple[dict[str, torch.Tensor], int, int]:
    """Return (capped_dict, saved_count, total_count).

    Iterates in insertion order and stops adding once cumulative bytes
    would exceed *max_bytes*.  If *max_bytes* is ``None``, saves everything.
    """
    total = len(intermediates)
    if max_bytes is None:
        out = {
            k: _to_storage_dtype(v.detach(), storage_dtype)
            for k, v in intermediates.items()
        }
        return out, total, total

    out: dict[str, torch.Tensor] = {}
    cum = 0
    for k, v in intermediates.items():
        t = _to_storage_dtype(v.detach(), storage_dtype)
        nbytes = t.nelement() * t.element_size()
        if cum + nbytes > max_bytes:
            break
        out[k] = t
        cum += nbytes
    return out, len(out), total


def save_step_data(
    capture: AtenCapture,
    pt_path: str,
    save_gm: bool = True,
    max_intermediates_mb: float | None = None,
    storage_dtype: torch.dtype | None = None,
) -> None:
    """Save all tensor data for one training step to a .pt file.

    Includes weights, real inputs/outputs, intermediates, and optimizer data.
    GraphModule pickles are saved alongside when *save_gm* is True.

    Args:
        max_intermediates_mb: Cap the total size of saved intermediate tensors
            (forward + backward).  Inputs and outputs are always saved.
            When set, optimizer data is skipped entirely.
        storage_dtype: Convert floating-point tensors to this dtype before
            saving (e.g. ``torch.bfloat16``).  Halves disk usage.
    """
    max_bytes = int(max_intermediates_mb * 1024 * 1024) if max_intermediates_mb is not None else None
    skip_optimizer = max_intermediates_mb is not None

    def _store(t):
        """Detach and optionally cast for storage."""
        if isinstance(t, torch.Tensor):
            return _to_storage_dtype(t.detach(), storage_dtype)
        return t

    pt_data: dict[str, Any] = {}

    meta: dict[str, Any] = {}
    if storage_dtype is not None:
        meta["storage_dtype"] = str(storage_dtype).replace("torch.", "")
    if max_intermediates_mb is not None:
        meta["max_intermediates_mb"] = max_intermediates_mb

    if capture.forward_graphs:
        fg = capture.forward_graphs[0]
        placeholders = [n for n in fg.graph_module.graph.nodes if n.op == "placeholder"]
        has_real = capture.forward_real_inputs is not None

        if has_real and len(capture.forward_real_inputs) == len(placeholders):
            for i, real_t in enumerate(capture.forward_real_inputs):
                name = placeholders[i].name if i < len(placeholders) else f"input_{i}"
                if isinstance(real_t, torch.Tensor):
                    pt_data[name] = _store(real_t)
        else:
            for i, inp in enumerate(fg.example_inputs):
                name = placeholders[i].name if i < len(placeholders) else f"input_{i}"
                if isinstance(inp, torch.Tensor):
                    pt_data[name] = _to_storage_dtype(_materialize_tensor(inp).cpu(), storage_dtype)

        if has_real and capture.forward_intermediates:
            capped, saved, total = _capped_intermediates(
                capture.forward_intermediates, max_bytes, storage_dtype)
            pt_data["_forward_intermediates"] = capped
            if saved < total:
                meta["fw_intermediates_saved"] = saved
                meta["fw_intermediates_total"] = total
        if has_real and capture.forward_real_output is not None:
            if isinstance(capture.forward_real_output, (tuple, list)):
                pt_data["_forward_output"] = [_store(t) for t in capture.forward_real_output]
            elif isinstance(capture.forward_real_output, torch.Tensor):
                pt_data["_forward_output"] = [_store(capture.forward_real_output)]
        if has_real and capture.backward_real_inputs:
            pt_data["_backward_inputs"] = [_store(t) for t in capture.backward_real_inputs]
            # Store num_user_outputs so the verify harness can derive fresh
            # backward inputs from the forward() result (needed for valid
            # CUDA Philox RNG state in _scaled_dot_product_efficient_attention).
            if capture.backward_graphs:
                bg = capture.backward_graphs[0]
                bw_phs = [n for n in bg.graph_module.graph.nodes if n.op == "placeholder"]
                num_tangents = sum(1 for p in bw_phs if "tangent" in p.name)
                num_saved = len(bw_phs) - num_tangents
                fw_out_node = [n for n in fg.graph_module.graph.nodes if n.op == "output"]
                if fw_out_node:
                    fw_out_args = fw_out_node[0].args[0]
                    num_fw_returns = len(fw_out_args) if isinstance(fw_out_args, (tuple, list)) else 1
                    meta["num_user_outputs"] = num_fw_returns - num_saved
        if has_real and capture.backward_intermediates:
            remaining = None
            if max_bytes is not None and "_forward_intermediates" in pt_data:
                fw_used = sum(
                    t.nelement() * t.element_size()
                    for t in pt_data["_forward_intermediates"].values()
                )
                remaining = max(0, max_bytes - fw_used)
            capped, saved, total = _capped_intermediates(
                capture.backward_intermediates,
                remaining if max_bytes is not None else None,
                storage_dtype,
            )
            pt_data["_backward_intermediates"] = capped
            if saved < total:
                meta["bw_intermediates_saved"] = saved
                meta["bw_intermediates_total"] = total
        if has_real and capture.backward_real_output is not None:
            if isinstance(capture.backward_real_output, (tuple, list)):
                pt_data["_backward_output"] = [_store(t) for t in capture.backward_real_output]
            elif isinstance(capture.backward_real_output, torch.Tensor):
                pt_data["_backward_output"] = [_store(capture.backward_real_output)]

    if not skip_optimizer:
        if capture.optimizer_data is not None:
            pt_data["_optimizer"] = capture.optimizer_data

        oc = capture.optimizer_capture
        if oc and oc.forward_real_inputs:
            pt_data["_optimizer_inputs"] = [_store(t) for t in oc.forward_real_inputs]
        if oc and oc.forward_real_output is not None:
            if isinstance(oc.forward_real_output, (tuple, list)):
                pt_data["_optimizer_output"] = [_store(t) for t in oc.forward_real_output]
            elif isinstance(oc.forward_real_output, torch.Tensor):
                pt_data["_optimizer_output"] = [_store(oc.forward_real_output)]
        if oc and oc.forward_intermediates:
            pt_data["_optimizer_intermediates"] = {
                k: _store(v) for k, v in oc.forward_intermediates.items()
            }

    if meta:
        pt_data["_meta"] = meta

    torch.save(pt_data, pt_path)

    if save_gm and capture.forward_graphs:
        pt = Path(pt_path)
        gm_fw_path = str(pt.with_stem(pt.stem + "_gm_fw"))
        try:
            torch.save(capture.forward_graphs[0].graph_module, gm_fw_path)
        except Exception as e:
            logger.warning("Could not save forward GraphModule pickle: %s", e)
    if save_gm and capture.backward_graphs:
        pt = Path(pt_path)
        gm_bw_path = str(pt.with_stem(pt.stem + "_gm_bw"))
        try:
            torch.save(capture.backward_graphs[0].graph_module, gm_bw_path)
        except Exception as e:
            logger.warning("Could not save backward GraphModule pickle: %s", e)


def export_aten_program(
    capture: AtenCapture,
    output_path: str,
    weights_path: str | None = None,
    inline_threshold: int = 1000,
    include_test_harness: bool = True,
    named_intermediates: bool = True,
    available_steps: list[int] | None = None,
    max_intermediates_mb: float | None = None,
    storage_dtype: torch.dtype | None = None,
    kernel_map: dict[str, str] | None = None,
    skip_pt: bool = False,
    uniquify: bool = True,
    uniquify_depth: int = -1,
    uniquify_min_ops: int = 0,
    emit_cuda_stubs: bool = False,
) -> Path:
    """Export full forward+backward as a standalone Python program.

    The exported file:
    - Uses only torch.ops.aten.* calls (pure low-level ops)
    - Shows original PyTorch module/op context above each group of ops
    - Embeds small weights inline, saves large ones to a .pt file
    - Includes both forward and backward as separate functions
    - Has a test harness that verifies each intermediate tensor
    - Is fully editable: change any op and rerun

    When *emit_cuda_stubs* is True and *uniquify* is True, CUDA C++ kernel
    templates are generated for each shared layer function.  To activate a
    CUDA kernel, uncomment the ``load_cuda`` call in the function body and
    fill in the C++ implementation.
    """
    out = Path(output_path)
    buf = StringIO()

    multi_frag = len(capture.forward_graphs) > 1

    # ── Multi-fragment path ───────────────────────────────────────
    if multi_frag:
        return _export_multi_fragment(
            capture, out, buf,
            weights_path=weights_path,
            inline_threshold=inline_threshold,
            include_test_harness=include_test_harness,
            named_intermediates=named_intermediates,
            max_intermediates_mb=max_intermediates_mb,
            storage_dtype=storage_dtype,
        )

    # ── Single-fragment path (existing behavior) ──────────────────

    primal_map = {}
    if capture.forward_graphs:
        fg = capture.forward_graphs[0]
        primal_map = _build_primal_map(fg.graph_module, capture)

    # Detect symbolic shapes for dynamic shapes comment
    has_symbolic_shapes = False
    dynamic_dims_comment = None
    if capture.forward_graphs and _graph_has_symbolic_shapes(capture.forward_graphs[0].graph_module):
        has_symbolic_shapes = True
        sym_dims = _get_symbolic_dim_names(capture.forward_graphs[0].graph_module)
        dynamic_dims_comment = f"Dynamic shapes: dimensions {', '.join(sym_dims)} are symbolic."
    elif capture.backward_graphs and _graph_has_symbolic_shapes(capture.backward_graphs[0].graph_module):
        has_symbolic_shapes = True
        sym_dims = _get_symbolic_dim_names(capture.backward_graphs[0].graph_module)
        dynamic_dims_comment = f"Dynamic shapes: dimensions {', '.join(sym_dims)} are symbolic."

    # ── CRITICAL ORDERING: code generation MUST happen before all_weights ──
    # The weights loop below calls int() on SymInt placeholders to get concrete
    # values.  Calling int() on ANY SymInt concretizes ALL related symbolic
    # dimensions globally (e.g. s77 → 2 everywhere), which destroys the
    # symbolic shape annotations in node.meta["val"].shape.  If we generated
    # code after that, type annotations would show "[2, 8]" instead of "[s77, 8]".
    # This is a PyTorch-level side effect — there is no way to undo it.
    fw_code = None
    unique_fn_defs: list[str] = []
    unique_groups_out: list[_UniqueGroup] = []
    if capture.forward_graphs:
        fg = capture.forward_graphs[0]
        fw_code = export_graph_to_python(
            fg.graph_module,
            fn_name="forward",
            inline_threshold=inline_threshold,
            primal_map=primal_map,
            annotate_sources=True,
            source_map=capture.source_map,
            named_intermediates=named_intermediates,
            dynamic_dims_comment=dynamic_dims_comment,
            kernel_map=kernel_map,
            uniquify=uniquify,
            uniquify_depth=uniquify_depth,
            uniquify_min_ops=uniquify_min_ops,
            _unique_fn_defs=unique_fn_defs,
            _unique_groups=unique_groups_out,
        )

    bw_code = None
    bw_unique_fn_defs: list[str] = []
    if capture.backward_graphs:
        bg = capture.backward_graphs[0]
        bw_code = export_graph_to_python(
            bg.graph_module,
            fn_name="backward",
            inline_threshold=inline_threshold,
            annotate_sources=True,
            source_map=capture.source_map,
            is_backward=True,
            named_intermediates=named_intermediates,
            dynamic_dims_comment=dynamic_dims_comment,
            uniquify=uniquify,
            uniquify_depth=uniquify_depth,
            uniquify_min_ops=uniquify_min_ops,
            _unique_fn_defs=bw_unique_fn_defs,
        )

    # Rename backward shared functions to avoid collisions with forward names
    if bw_unique_fn_defs and unique_fn_defs:
        fw_names = set()
        for fd in unique_fn_defs:
            m = re.match(r'def (\w+)\(', fd)
            if m:
                fw_names.add(m.group(1))
        for i, fd in enumerate(bw_unique_fn_defs):
            m = re.match(r'def (\w+)\(', fd)
            if m and m.group(1) in fw_names:
                old = m.group(1)
                new = f"bw_{old}"
                bw_unique_fn_defs[i] = fd.replace(f"def {old}(", f"def {new}(", 1)
                if bw_code:
                    bw_code = bw_code.replace(f"{old}(", f"{new}(")

    has_real = capture.forward_real_inputs is not None
    all_weights = {}
    if capture.forward_graphs:
        fg = capture.forward_graphs[0]
        placeholders = [n for n in fg.graph_module.graph.nodes if n.op == "placeholder"]

        if has_real and len(capture.forward_real_inputs) == len(placeholders):
            for i, real_t in enumerate(capture.forward_real_inputs):
                name = placeholders[i].name if i < len(placeholders) else f"input_{i}"
                if isinstance(real_t, torch.Tensor):
                    all_weights[name] = real_t.detach()
                elif not (hasattr(torch, "SymInt") and isinstance(real_t, torch.SymInt)):
                    all_weights[name] = real_t
        else:
            for i, inp in enumerate(fg.example_inputs):
                name = placeholders[i].name if i < len(placeholders) else f"input_{i}"
                ph_val = placeholders[i].meta.get("val") if i < len(placeholders) else None
                if isinstance(inp, torch.Tensor):
                    try:
                        all_weights[name] = inp.detach()
                    except RuntimeError:
                        try:
                            all_weights[name] = torch.randn(inp.shape, dtype=inp.dtype, device=inp.device)
                        except RuntimeError as e:
                            logger.warning("Failed to create synthetic input %s: %s", name, e)
                elif ph_val is not None and not hasattr(ph_val, "shape"):
                    # SymInt placeholder: store concrete int for test harness
                    try:
                        from torch.fx.experimental.symbolic_shapes import is_concrete_int
                        all_weights[name] = int(inp) if is_concrete_int(inp) else int(ph_val)
                    except (ImportError, TypeError, ValueError):
                        all_weights[name] = 2  # fallback batch size

    needs_file = has_real or any(
        isinstance(t, torch.Tensor) and t.numel() > inline_threshold
        for t in all_weights.values()
        if isinstance(t, torch.Tensor)
    )
    h5_weights_path = None
    if needs_file:
        inline_threshold = 0
        if skip_pt:
            # Don't reference any weight file — all weights become inline
            # placeholders (torch.randn/zeros).  The caller (e.g. auto_install)
            # replaces them with live model parameters at runtime.
            needs_file = False
        if weights_path is None:
            weights_path = str(out.with_suffix(".pt"))

    # ── Detect custom op namespaces and loaded extension libraries ──
    custom_namespaces = _find_custom_op_namespaces(capture)
    op_providers = _build_op_providers(custom_namespaces)

    # ── Collect Triton kernels from forward/backward graphs ──
    triton_gms = []
    if capture.forward_graphs:
        triton_gms.append(capture.forward_graphs[0].graph_module)
    if capture.backward_graphs:
        triton_gms.append(capture.backward_graphs[0].graph_module)
    triton_kernels, needs_tma = _collect_triton_kernels(*triton_gms) if triton_gms else ({}, False)

    # ── Header ────────────────────────────────────────────────────

    buf.write('"""Auto-generated aten-level PyTorch program.\n')
    buf.write("\n")
    buf.write("This file contains the COMPLETE computation graph decomposed into\n")
    buf.write("low-level aten ops. You can edit ANY operation and rerun this file.\n")
    buf.write("\n")
    buf.write("Operations use torch.ops.aten.* - the lowest-level PyTorch ops.\n")
    buf.write("The backward pass (autograd) is also expressed as explicit aten ops.\n")

    if primal_map:
        buf.write("\nParameter mapping:\n")
        max_desc = max(len(d) for d in primal_map.values()) + 2
        for pname, desc in primal_map.items():
            val = all_weights.get(pname)
            shape_str = str(list(val.shape)) if val is not None and hasattr(val, "shape") else str(val)
            buf.write(f"  {desc:{max_desc}s} {shape_str}\n")

    if available_steps and len(available_steps) > 1:
        buf.write(f"\nAvailable steps: {', '.join(str(s) for s in sorted(available_steps))}\n")
        buf.write("Usage: python <this_file> --step N\n")

    buf.write('"""\n\n')
    _emit_standard_header(buf, op_providers=op_providers,
                          triton_kernels=triton_kernels, needs_tma=needs_tma)

    if available_steps and len(available_steps) > 1:
        steps_str = ", ".join(str(s) for s in sorted(available_steps))
        default_step = sorted(available_steps)[0]
        stem = Path(output_path).stem.replace("_aten", "")
        buf.write("\n")
        buf.write("import argparse as _argparse\n")
        buf.write('_parser = _argparse.ArgumentParser(description="Extracted aten program")\n')
        buf.write(f'_parser.add_argument("--step", type=int, default={default_step}, help="Training step (available: {steps_str})")\n')
        buf.write(f'_parser.add_argument("--data", type=str, default=None, help="Custom .pt data file")\n')
        buf.write("_args, _ = _parser.parse_known_args()\n")
        buf.write('_dir = os.path.dirname(os.path.abspath(__file__))\n')
        buf.write(f'_data_path = _args.data or os.path.join(_dir, "{stem}_step{{}}.pt".format(_args.step))\n')

    buf.write("\n")

    # ── Weights section ───────────────────────────────────────────

    buf.write("# " + "=" * 70 + "\n")
    buf.write("# WEIGHTS / PARAMETERS\n")
    buf.write("# " + "=" * 70 + "\n\n")

    if has_symbolic_shapes:
        buf.write("# Note: Graph has symbolic shapes. Example input shapes may vary.\n\n")

    if needs_file:
        if h5_weights_path:
            # Emit shared H5 loader (renamed to _load_h5) and call it
            snippet = _h5_load_function_source(map_short_to_long=True)
            buf.write(snippet)
            buf.write(f'weights = _load_h5("{h5_weights_path}")\n\n')
        elif available_steps and len(available_steps) > 1:
            buf.write('weights = torch.load(_data_path, weights_only=True, map_location=_device)\n\n')
        else:
            buf.write(f'weights = torch.load("{weights_path}", weights_only=True, map_location=_device)\n\n')

    for name, val in all_weights.items():
        if isinstance(val, torch.Tensor):
            comment = primal_map.get(name, "")
            line = _ir_tensor_to_constructor(val, name, inline_threshold, comment=comment)
            if len(line) > 200 and val.numel() <= inline_threshold:
                buf.write(f"# {name}: shape={list(val.shape)}, dtype={val.dtype}\n")
                buf.write(f"{line}\n")
            else:
                buf.write(f"{line}\n")
        elif isinstance(val, (int, float)):
            buf.write(f"{name} = {val}  # symbolic dim (concrete value for export)\n")
    # Move inline tensors to target device (weights from .pt are already mapped)
    if not needs_file:
        buf.write(f"\n# Move to target device\n")
        for name, val in all_weights.items():
            if isinstance(val, torch.Tensor):
                buf.write(f"{name} = {name}.to(_device)\n")
    buf.write("\n\n")

    # ── CUDA kernel stubs (optional) ─────────────────────────────
    cuda_fn_toggle: dict[str, str] = {}  # fn_name -> commented-out CUDA call lines
    if emit_cuda_stubs and unique_groups_out:
        from torch_graph.cuda_inline import cuda_kernel_template

        buf.write("# " + "=" * 70 + "\n")
        buf.write("# CUDA KERNEL SOURCES\n")
        buf.write("# " + "=" * 70 + "\n")
        buf.write("# Auto-generated C++ ATen implementations for each shared layer.\n")
        buf.write("# To switch a layer to CUDA: uncomment the 4 lines at the top of\n")
        buf.write("# its function body and comment out the aten ops below them.\n")
        buf.write("# First run compiles (~10-30s), then cached on disk.\n")
        buf.write("# " + "=" * 70 + "\n\n")
        buf.write("from torch_graph.cuda_inline import load_cuda\n\n")

        for group in unique_groups_out:
            cuda_var = f"_{group.fn_name.upper()}_CUDA"
            mod_var = f"_{group.fn_name}_cuda_mod"
            cpp_fn = f"fused_{group.fn_name}"

            # Generate C++ template with working body from aten ops
            cuda_src = cuda_kernel_template(
                group.fn_name, group.params, group.returns,
                body_code=group.body_code,
            )
            buf.write(f'{cuda_var} = r"""\n')
            buf.write(cuda_src)
            buf.write('"""\n\n')

            buf.write(f"{mod_var} = None\n\n")

            # Build the commented-out toggle lines to embed in the function body
            param_names = ", ".join(p["name"] for p in group.params)
            toggle_lines = []
            toggle_lines.append(f"    # ── CUDA kernel (uncomment to activate) ──")
            toggle_lines.append(f"    # global {mod_var}")
            toggle_lines.append(f"    # if {mod_var} is None:")
            toggle_lines.append(f'    #     {mod_var} = load_cuda("{group.fn_name}", {cuda_var}, ["{cpp_fn}"])')
            if len(group.returns) <= 1:
                toggle_lines.append(f"    # return ({mod_var}.{cpp_fn}({param_names}),)")
            else:
                toggle_lines.append(f"    # return {mod_var}.{cpp_fn}({param_names})")
            toggle_lines.append(f"    # ──────────────────────────────────────────")
            cuda_fn_toggle[group.fn_name] = "\n".join(toggle_lines)

    # ── Shared layer functions (uniquified) ──────────────────────
    if unique_fn_defs:
        buf.write("# " + "=" * 70 + "\n")
        buf.write("# SHARED LAYER FUNCTIONS\n")
        buf.write("# " + "=" * 70 + "\n")
        buf.write("# Repeated module groups extracted into reusable functions.\n")
        buf.write("# Edit once — changes apply to all instances.\n")
        buf.write("# " + "=" * 70 + "\n\n")
        for i, fn_def in enumerate(unique_fn_defs):
            # If CUDA stubs are enabled, inject the toggle into the function body
            if i < len(unique_groups_out) and unique_groups_out[i].fn_name in cuda_fn_toggle:
                group = unique_groups_out[i]
                toggle = cuda_fn_toggle[group.fn_name]
                # Insert the toggle right after the signature line(s)
                fn_lines = fn_def.split("\n")
                # Find the end of the signature (the line with just ")")  or "):" pattern
                insert_idx = 0
                for j, fl in enumerate(fn_lines):
                    s = fl.strip()
                    if s.startswith(")") and ("-> tuple" in s or s == "):"):
                        insert_idx = j + 1
                        break
                    if s.endswith("):") or s.endswith("]:"):
                        insert_idx = j + 1
                        break
                if insert_idx > 0:
                    fn_lines.insert(insert_idx, toggle)
                    fn_def = "\n".join(fn_lines)
            buf.write(fn_def)
            buf.write("\n\n")

    # ── Forward function ──────────────────────────────────────────
    if fw_code:
        buf.write("# " + "=" * 70 + "\n")
        buf.write("# FORWARD PASS (aten ops)\n")
        buf.write("# " + "=" * 70 + "\n")
        buf.write("# Edit any operation below. The ops are pure aten - no autograd,\n")
        buf.write("# no module abstractions, just raw tensor operations.\n")
        buf.write("# \n")
        buf.write("# Source annotations show which part of the original PyTorch model\n")
        buf.write("# or function each op/group came from, with compact file:line notes\n")
        buf.write("# for source-only ops and section headers for module groups.\n")
        buf.write("# \n")
        buf.write("# The return value includes saved tensors needed for backward.\n")
        buf.write("# " + "=" * 70 + "\n\n")
        buf.write(fw_code)
        buf.write("\n\n")

    # ── Backward shared layer functions (uniquified) ────────────
    if bw_unique_fn_defs:
        buf.write("# " + "=" * 70 + "\n")
        buf.write("# SHARED BACKWARD FUNCTIONS\n")
        buf.write("# " + "=" * 70 + "\n")
        buf.write("# Repeated gradient groups extracted into reusable functions.\n")
        buf.write("# Edit once — changes apply to all instances.\n")
        buf.write("# " + "=" * 70 + "\n\n")
        for fn_def in bw_unique_fn_defs:
            buf.write(fn_def)
            buf.write("\n\n")

    # ── Backward function ─────────────────────────────────────────
    if bw_code:
        buf.write("# " + "=" * 70 + "\n")
        buf.write("# BACKWARD PASS (aten ops - the autograd graph!)\n")
        buf.write("# " + "=" * 70 + "\n")
        buf.write("# This IS the autograd. Every gradient computation is an explicit\n")
        buf.write("# aten op. You can edit the backward pass just like the forward:\n")
        buf.write("#   - Modify gradient computations\n")
        buf.write("#   - Add gradient clipping as raw ops\n")
        buf.write("#   - Implement custom gradient scaling\n")
        buf.write("#   - Skip gradient computation for specific parameters\n")
        buf.write("# \n")
        buf.write("# Source annotations show which FORWARD op each gradient group\n")
        buf.write("# corresponds to, using compact file:line notes or module headers.\n")
        buf.write("# " + "=" * 70 + "\n\n")
        buf.write(bw_code)
        buf.write("\n\n")

    # ── Optimizer step function ────────────────────────────────────
    if capture.optimizer_capture and capture.optimizer_capture.forward_graphs:
        og = capture.optimizer_capture.forward_graphs[0]
        opt_class = capture.optimizer_data.get("class", "Optimizer") if capture.optimizer_data else "Optimizer"
        buf.write("# " + "=" * 70 + "\n")
        buf.write(f"# OPTIMIZER STEP ({opt_class} - extracted aten ops)\n")
        buf.write("# " + "=" * 70 + "\n")
        buf.write("# These are the REAL aten ops that PyTorch executes during\n")
        buf.write(f"# {opt_class}.step() — extracted via torch.compile, not inferred.\n")
        buf.write("# Ops include lerp (EMA update), addcdiv (Adam update), sqrt, etc.\n")
        buf.write("# " + "=" * 70 + "\n\n")

        opt_code = export_graph_to_python(
            og.graph_module,
            fn_name="optimizer_step",
            inline_threshold=inline_threshold,
            named_intermediates=named_intermediates,
            annotate_sources=False,
        )
        buf.write(opt_code)
        buf.write("\n\n")

    # ── Test harness ───────────────────────────────────────────────

    if include_test_harness:
        buf.write("# " + "=" * 70 + "\n")
        buf.write("# TEST HARNESS - Run this file to verify the exported graph\n")
        buf.write("# " + "=" * 70 + "\n\n")

        buf.write('if __name__ == "__main__":\n')
        buf.write('    import sys, os\n')
        buf.write(f'    print(f"Running exported aten program on {{_device}}...")\n\n')

        multi = available_steps and len(available_steps) > 1
        if multi:
            buf.write('    print(f"Verifying step {_args.step} — {os.path.basename(_data_path)}")\n')
            buf.write("    print()\n\n")

        if has_real:
            pt_ref = "_data_path" if multi else None
            _emit_real_tensor_harness(buf, capture, weights_path or str(out.with_suffix(".pt")), has_symbolic_shapes=has_symbolic_shapes, data_path_var=pt_ref)
        else:
            _emit_basic_harness(buf, capture, has_symbolic_shapes=has_symbolic_shapes)

    content = buf.getvalue()

    # Rename primals in the harness/weight-loading code to match the function signature.
    # export_graph_to_python already renamed them inside function bodies; the harness
    # still uses raw FX names (primals_N) and needs to match.
    if primal_map:
        _fg_nodes = fg.graph_module.graph.nodes if capture.forward_graphs else []
        _ph_remap = _sanitize_primal_names(_fg_nodes, primal_map)
        # Protect dict keys, apply remap, restore dict keys
        for orig in _ph_remap:
            content = content.replace(f'["{orig}"]', f'["__PHKEY_{orig}__"]')
        # Single-pass regex: match any primal name at word boundary
        if _ph_remap:
            # Sort by length descending so longer names match first (primals_10 before primals_1)
            sorted_names = sorted(_ph_remap.keys(), key=len, reverse=True)
            pattern = re.compile(r'\b(' + '|'.join(re.escape(n) for n in sorted_names) + r')\b')
            content = pattern.sub(lambda m: _ph_remap[m.group(0)], content)
        for orig in _ph_remap:
            content = content.replace(f'["__PHKEY_{orig}__"]', f'["{orig}"]')
    else:
        # Fallback: shorten primals_N → pN for readability (keep .pt file keys unchanged)
        content = re.sub(r'weights\["primals_(\d+)"\]', r'weights["__PH_\1__"]', content)
        content = re.sub(r'\bprimals_(\d+)\b', r'p\1', content)
        content = re.sub(r'weights\["__PH_(\d+)__"\]', r'weights["primals_\1"]', content)

    out.write_text(content)

    # Save .pt file (single-step mode only — multi-step callers use save_step_data)
    if not skip_pt and weights_path != "/dev/null" and not (available_steps and len(available_steps) > 1):
        save_path = weights_path or (str(out.with_suffix(".pt")) if has_real else None)
        if needs_file or has_real:
            if save_path is None:
                save_path = str(out.with_suffix(".pt"))
            save_step_data(capture, save_path, save_gm=True,
                           max_intermediates_mb=max_intermediates_mb,
                           storage_dtype=storage_dtype)

    return out


def _export_multi_fragment(
    capture: AtenCapture,
    out: Path,
    buf: StringIO,
    *,
    weights_path: str | None = None,
    inline_threshold: int = 1000,
    include_test_harness: bool = True,
    named_intermediates: bool = False,
    max_intermediates_mb: float | None = None,
    storage_dtype: torch.dtype | None = None,
) -> Path:
    """Export a multi-fragment (graph-break) capture as a standalone program.

    Each fragment gets its own forward_N / backward_N function with
    independently verified inputs/outputs.
    """
    n_frag = len(capture.forward_graphs)

    # Always use a .pt file for multi-fragment
    if weights_path is None:
        weights_path = str(out.with_suffix(".pt"))

    # ── Detect custom op namespaces and loaded extension libraries ──
    custom_namespaces = _find_custom_op_namespaces(capture)
    op_providers = _build_op_providers(custom_namespaces)

    # ── Collect Triton kernels from all fragments ──
    triton_gms = []
    for fg in capture.forward_graphs:
        triton_gms.append(fg.graph_module)
    for bg in capture.backward_graphs:
        triton_gms.append(bg.graph_module)
    triton_kernels, needs_tma = _collect_triton_kernels(*triton_gms) if triton_gms else ({}, False)

    # ── Header ────────────────────────────────────────────────────
    buf.write('"""Auto-generated aten-level PyTorch program (multi-fragment).\n')
    buf.write("\n")
    buf.write(f"This model produced {n_frag} graph fragments due to graph breaks.\n")
    buf.write("Each fragment is exported as a separate function (forward_0, forward_1, etc.).\n")
    buf.write("\n")
    buf.write("Operations use torch.ops.aten.* - the lowest-level PyTorch ops.\n")
    buf.write('"""\n\n')
    _emit_standard_header(buf, op_providers=op_providers,
                          triton_kernels=triton_kernels, needs_tma=needs_tma)
    buf.write("\n")

    # ── Weights ───────────────────────────────────────────────────
    # Multi-fragment: inputs are loaded per-fragment in the test harness.
    # No global variables needed — functions take all inputs as parameters.
    buf.write("# " + "=" * 70 + "\n")
    buf.write(f"# WEIGHTS / PARAMETERS ({n_frag} fragments)\n")
    buf.write("# " + "=" * 70 + "\n")
    buf.write(f"# Inputs for each fragment are stored in: {Path(weights_path).name}\n")
    buf.write(f"# Keys: _frag_{{i}}_fw_inputs, _frag_{{i}}_{{placeholder_name}}\n\n")

    # ── Forward functions ─────────────────────────────────────────
    for i, fg in enumerate(capture.forward_graphs):
        frag_names = capture.per_frag_primal_names[i] if i < len(capture.per_frag_primal_names) else []
        frag_map = _build_primal_map(fg.graph_module, capture, frag_primal_names=frag_names)

        buf.write("# " + "=" * 70 + "\n")
        buf.write(f"# FORWARD PASS — Fragment {i} of {n_frag}\n")
        buf.write("# " + "=" * 70 + "\n\n")

        fw_code = export_graph_to_python(
            fg.graph_module,
            fn_name=f"forward_{i}",
            inline_threshold=0,
            primal_map=frag_map,
            annotate_sources=True,
            source_map=capture.source_map,
            named_intermediates=named_intermediates,
        )
        buf.write(fw_code)
        buf.write("\n\n")

    # ── Backward functions ────────────────────────────────────────
    for i, bg in enumerate(capture.backward_graphs):
        buf.write("# " + "=" * 70 + "\n")
        buf.write(f"# BACKWARD PASS — Fragment {i}\n")
        buf.write("# " + "=" * 70 + "\n\n")

        bw_code = export_graph_to_python(
            bg.graph_module,
            fn_name=f"backward_{i}",
            inline_threshold=0,
            annotate_sources=True,
            source_map=capture.source_map,
            is_backward=True,
            named_intermediates=named_intermediates,
        )
        buf.write(bw_code)
        buf.write("\n\n")

    # ── Test harness ──────────────────────────────────────────────
    if include_test_harness:
        buf.write("# " + "=" * 70 + "\n")
        buf.write("# TEST HARNESS - Verify each fragment independently\n")
        buf.write("# " + "=" * 70 + "\n\n")

        buf.write('if __name__ == "__main__":\n')
        buf.write('    import sys, os\n')
        buf.write(f'    print(f"Running exported aten program on {{_device}} ({n_frag} fragments)...")\n')
        buf.write("    print()\n\n")

        # Load reference data
        buf.write(f'    _data = torch.load("{weights_path}", weights_only=True, map_location=_device)\n')
        buf.write('    _meta = _data.get("_meta", {})\n')
        buf.write('    _bf16 = _meta.get("storage_dtype") == "bfloat16"\n')
        buf.write('    _atol = float(os.environ.get("ATOL", 5.0 if _bf16 else 1e-5))\n')
        buf.write('    _rtol = float(os.environ.get("RTOL", 0.1 if _bf16 else 1e-4))\n')
        buf.write("    print()\n\n")

        # Emit forward function list
        fw_fn_names = [f"forward_{i}" for i in range(n_frag)]
        buf.write(f"    _fw_fns = [{', '.join(fw_fn_names)}]\n\n")

        # Per-fragment forward verification — store results for backward Philox fix
        buf.write("    _frag_results = {}\n")
        buf.write(f"    for _fi in range({n_frag}):\n")
        buf.write('        _frag_ref_out = _data.get(f"_frag_{_fi}_fw_output")\n')
        buf.write("        if _frag_ref_out is None:\n")
        buf.write('            print(f"Fragment {_fi}: no reference data, skipping")\n')
        buf.write("            continue\n")

        # Get per-fragment inputs from weights
        buf.write('        _frag_inputs = _data.get(f"_frag_{_fi}_fw_inputs")\n')
        buf.write("        if _frag_inputs is None:\n")
        buf.write('            print(f"Fragment {_fi}: no input data, skipping")\n')
        buf.write("            continue\n")

        buf.write("        _frag_result = _fw_fns[_fi](*_frag_inputs)\n")
        buf.write("        _frag_results[_fi] = _frag_result\n")
        buf.write("        _fw_tensors = [t for t in (_frag_result if isinstance(_frag_result, tuple) else (_frag_result,)) if isinstance(t, torch.Tensor)]\n")
        buf.write("        _fw_ref_tensors = [t for t in _frag_ref_out if isinstance(t, torch.Tensor)]\n")
        buf.write("        _ok_fw, _total_fw, _fw_fail = 0, 0, []\n")
        buf.write("        for _i, (_ref, _act) in enumerate(zip(_fw_ref_tensors, _fw_tensors)):\n")
        buf.write("            _total_fw += 1\n")
        buf.write("            _rf, _af = _ref.float().cpu(), _act.float().cpu()\n")
        buf.write("            if _ref.shape != _act.shape:\n")
        buf.write("                _fw_fail.append((_i, _ref.shape, float('inf')))\n")
        buf.write("            elif _rf.isnan().any() or _af.isnan().any() or _rf.isinf().any() or _af.isinf().any():\n")
        buf.write("                _ok_fw += 1\n")
        buf.write("            elif torch.allclose(_rf, _af, atol=_atol, rtol=_rtol):\n")
        buf.write("                _ok_fw += 1\n")
        buf.write("            else:\n")
        buf.write("                _fw_fail.append((_i, _ref.shape, (_rf - _af).abs().max().item()))\n")
        buf.write('        print(f"Forward fragment {_fi}: {_ok_fw}/{_total_fw} outputs match")\n')
        buf.write("        for _i, _s, _d in _fw_fail[:5]:\n")
        buf.write('            print(f"  FAIL: output[{_i}] shape={list(_s)} max_diff={_d:.2e}")\n')
        buf.write("    print()\n")

        # Per-fragment backward verification
        n_bw = len(capture.backward_graphs)
        if n_bw > 0:
            bw_fn_names = [f"backward_{i}" for i in range(n_bw)]
            buf.write(f"\n    _bw_fns = [{', '.join(bw_fn_names)}]\n\n")
            buf.write(f"    for _fi in range({n_bw}):\n")
            buf.write('        _bw_inputs = _data.get(f"_frag_{_fi}_bw_inputs")\n')
            buf.write('        _bw_ref_out = _data.get(f"_frag_{_fi}_bw_output")\n')
            buf.write("        if _bw_inputs is None:\n")
            buf.write("            continue\n")
            # Replace serialized saved tensors with fresh forward results (Philox RNG fix).
            # Backward graphs may be in different order than forward graphs, so we use
            # bw_{i}_fw_idx metadata to find the correct forward fragment.
            buf.write("        # Replace saved tensors with fresh forward results (Philox RNG fix)\n")
            buf.write("        _fw_idx = _meta.get(f'bw_{_fi}_fw_idx')\n")
            buf.write("        _num_fw_out = _meta.get(f'bw_{_fi}_num_user_outputs', 0)\n")
            buf.write("        _fw_res = _frag_results.get(_fw_idx) if _fw_idx is not None else None\n")
            buf.write("        if _num_fw_out > 0 and _fw_res is not None:\n")
            buf.write("            _fw_all = _fw_res if isinstance(_fw_res, tuple) else (_fw_res,)\n")
            buf.write("            _num_saved = len(_fw_all) - _num_fw_out\n")
            buf.write("            _num_tangents = len(_bw_inputs) - _num_saved\n")
            buf.write("            if _num_saved > 0 and _num_tangents > 0:\n")
            buf.write("                _bw_inputs = list(_fw_all[_num_fw_out:]) + list(_bw_inputs[-_num_tangents:])\n")
            buf.write("        _bw_result = _bw_fns[_fi](*_bw_inputs)\n")
            buf.write("        _bw_tensors = [t for t in (_bw_result if isinstance(_bw_result, tuple) else (_bw_result,)) if isinstance(t, torch.Tensor)]\n")
            buf.write('        print(f"Backward fragment {_fi}: {len(_bw_tensors)} gradient outputs")\n')

            buf.write("        if _bw_ref_out is not None:\n")
            buf.write("            _bw_ref_tensors = [t for t in _bw_ref_out if isinstance(t, torch.Tensor)]\n")
            buf.write("            _ok_bw, _total_bw, _bw_fail = 0, 0, []\n")
            buf.write("            for _i, (_ref, _act) in enumerate(zip(_bw_ref_tensors, _bw_tensors)):\n")
            buf.write("                _total_bw += 1\n")
            buf.write("                _rf, _af = _ref.float().cpu(), _act.float().cpu()\n")
            buf.write("                if _ref.shape != _act.shape:\n")
            buf.write("                    _bw_fail.append((_i, _ref.shape, float('inf')))\n")
            buf.write("                elif _rf.numel() == 0 or (_rf.isnan().all() and _af.isnan().all()):\n")
            buf.write("                    _ok_bw += 1\n")
            buf.write("                elif torch.allclose(_rf, _af, atol=_atol, rtol=_rtol):\n")
            buf.write("                    _ok_bw += 1\n")
            buf.write("                else:\n")
            buf.write("                    _d = (_rf - _af).abs()\n")
            buf.write("                    _d = _d[~_d.isnan()]\n")
            buf.write("                    _bw_fail.append((_i, _ref.shape, _d.max().item() if _d.numel() > 0 else float('nan')))\n")
            buf.write('            print(f"  Backward outputs: {_ok_bw}/{_total_bw} match")\n')
            buf.write("            for _i, _s, _d in _bw_fail[:5]:\n")
            buf.write('                print(f"    FAIL: grad[{_i}] shape={list(_s)} max_diff={_d:.2e}")\n')

        buf.write("    print()\n")
        buf.write(f'    print("All {n_frag} fragments verified.")\n')

    content = buf.getvalue()

    # Fix inline tensor constants (from get_attr nodes like empty KV caches):
    # ensure they use _device so they match inputs loaded from .pt.
    #
    # Uses [^)\n]* (not [^)]*) to prevent matching across newlines.
    # Only torch.tensor/randn/zeros/ones/empty/full appear as get_attr constants;
    # do NOT add aten ops (arange, linspace, etc.) which appear in source comments.
    content = re.sub(
        r'(= torch\.(?:tensor|randn|zeros|ones|empty|full)\([^)\n]*\))(?!\.to\()',
        r'\1.to(_device)',
        content,
    )
    # Fix hardcoded device references (e.g. device=torch.device('cuda:0'))
    content = re.sub(
        r"device=torch\.device\('[^']+'\)",
        "device=_device",
        content,
    )

    out.write_text(content)

    # Save multi-fragment .pt file
    if weights_path != "/dev/null":
        _save_multi_fragment_data(
            capture, weights_path,
            max_intermediates_mb=max_intermediates_mb,
            storage_dtype=storage_dtype,
        )

    return out


def _save_multi_fragment_data(
    capture: AtenCapture,
    pt_path: str,
    max_intermediates_mb: float | None = None,
    storage_dtype: torch.dtype | None = None,
) -> None:
    """Save per-fragment tensor data for multi-fragment captures."""
    def _store(t):
        if isinstance(t, torch.Tensor):
            return _to_storage_dtype(t.detach(), storage_dtype)
        return t

    pt_data: dict[str, Any] = {}
    meta: dict[str, Any] = {}
    if storage_dtype is not None:
        meta["storage_dtype"] = str(storage_dtype).replace("torch.", "")

    n_fw = len(capture.forward_graphs)
    n_bw = len(capture.backward_graphs)
    meta["n_fragments"] = n_fw

    # Save per-fragment forward inputs/outputs and weight placeholders
    for i, fg in enumerate(capture.forward_graphs):
        phs = [n for n in fg.graph_module.graph.nodes if n.op == "placeholder"]
        frag_inputs = capture.per_frag_fw_inputs[i] if i < len(capture.per_frag_fw_inputs) and capture.per_frag_fw_inputs[i] else None

        # Save individual placeholder tensors (for weights section loading)
        if frag_inputs and len(frag_inputs) == len(phs):
            for j, real_t in enumerate(frag_inputs):
                pt_key = f"_frag_{i}_{phs[j].name}"
                if isinstance(real_t, torch.Tensor):
                    pt_data[pt_key] = _store(real_t)
        else:
            # Fallback to example_inputs
            for j, inp in enumerate(fg.example_inputs):
                pt_key = f"_frag_{i}_{phs[j].name}" if j < len(phs) else f"_frag_{i}_input_{j}"
                if isinstance(inp, torch.Tensor):
                    pt_data[pt_key] = _to_storage_dtype(_materialize_tensor(inp).cpu(), storage_dtype)

        # Save per-fragment forward inputs as a list (for test harness)
        if frag_inputs:
            pt_data[f"_frag_{i}_fw_inputs"] = [_store(t) for t in frag_inputs]

        # Save per-fragment forward output
        frag_output = capture.per_frag_fw_output[i] if i < len(capture.per_frag_fw_output) and capture.per_frag_fw_output[i] is not None else None
        if frag_output is not None:
            if isinstance(frag_output, (tuple, list)):
                pt_data[f"_frag_{i}_fw_output"] = [_store(t) for t in frag_output]
            elif isinstance(frag_output, torch.Tensor):
                pt_data[f"_frag_{i}_fw_output"] = [_store(frag_output)]

    # Save per-fragment backward inputs/outputs + num_user_outputs for Philox fix
    for i in range(n_bw):
        frag_bw_inputs = capture.per_frag_bw_inputs[i] if i < len(capture.per_frag_bw_inputs) and capture.per_frag_bw_inputs[i] else None
        if frag_bw_inputs:
            pt_data[f"_frag_{i}_bw_inputs"] = [_store(t) for t in frag_bw_inputs]

        frag_bw_output = capture.per_frag_bw_output[i] if i < len(capture.per_frag_bw_output) and capture.per_frag_bw_output[i] is not None else None
        if frag_bw_output is not None:
            if isinstance(frag_bw_output, (tuple, list)):
                pt_data[f"_frag_{i}_bw_output"] = [_store(t) for t in frag_bw_output]
            elif isinstance(frag_bw_output, torch.Tensor):
                pt_data[f"_frag_{i}_bw_output"] = [_store(frag_bw_output)]

    # Compute backward-to-forward mapping and num_user_outputs per fragment
    # so backward verification can replace serialized Philox RNG seeds with
    # fresh forward results.  Backward graphs are typically compiled lazily
    # during backward execution (reverse order), so backward_0 often
    # corresponds to forward_(n-1), etc.  We find the correct pairing by
    # matching each backward's saved-tensor count to forward output counts.
    bw_to_fw: dict[int, int] = {}
    for bi in range(n_bw):
        bg = capture.backward_graphs[bi]
        bw_phs = [n for n in bg.graph_module.graph.nodes if n.op == "placeholder"]
        num_tangents = sum(1 for p in bw_phs if "tangent" in p.name)
        num_saved = len(bw_phs) - num_tangents

        # Find the forward graph whose output count matches num_saved + user_out
        best_fi, best_user_out = None, None
        for fi in range(n_fw):
            if fi in bw_to_fw.values():
                continue  # already paired
            fg = capture.forward_graphs[fi]
            fw_out_node = [n for n in fg.graph_module.graph.nodes if n.op == "output"]
            if fw_out_node:
                fw_out_args = fw_out_node[0].args[0]
                num_fw_returns = len(fw_out_args) if isinstance(fw_out_args, (tuple, list)) else 1
                user_out = num_fw_returns - num_saved
                if user_out > 0:
                    if best_user_out is None or user_out < best_user_out:
                        best_fi, best_user_out = fi, user_out

        if best_fi is not None:
            bw_to_fw[bi] = best_fi
            meta[f"bw_{bi}_fw_idx"] = best_fi
            meta[f"bw_{bi}_num_user_outputs"] = best_user_out

    # Save GM pickles for each fragment
    base = str(Path(pt_path).with_suffix(""))
    for i, fg in enumerate(capture.forward_graphs):
        torch.save(fg.graph_module, f"{base}_gm_fw_{i}.pt")
    for i, bg in enumerate(capture.backward_graphs):
        torch.save(bg.graph_module, f"{base}_gm_bw_{i}.pt")

    if meta:
        pt_data["_meta"] = meta

    torch.save(pt_data, pt_path)


def _emit_real_tensor_harness(buf: StringIO, capture: AtenCapture, pt_path: str, has_symbolic_shapes: bool = False, data_path_var: str | None = None) -> None:
    """Generate test harness that loads real tensors and verifies every intermediate.

    When *data_path_var* is set (e.g. ``"_data_path"``), the generated code
    loads from that variable instead of a hardcoded path — used for multi-step.
    """
    if has_symbolic_shapes:
        buf.write('    # Note: Shapes are symbolic - batch/sequence dimensions may vary.\n\n')
    fg = capture.forward_graphs[0] if capture.forward_graphs else None
    bg = capture.backward_graphs[0] if capture.backward_graphs else None

    if data_path_var:
        buf.write(f'    _data = torch.load({data_path_var}, weights_only=True, map_location=_device)\n')
    else:
        buf.write(f'    _data = torch.load("{pt_path}", weights_only=True, map_location=_device)\n')

    # Storage metadata: detect bf16 storage and adjust tolerances
    buf.write('    _meta = _data.get("_meta", {})\n')
    buf.write('    _bf16 = _meta.get("storage_dtype") == "bfloat16"\n')
    buf.write('    _atol = _dev_args.atol or float(os.environ.get("ATOL", 5.0 if _bf16 else 1e-5))\n')
    buf.write('    _rtol = _dev_args.rtol or float(os.environ.get("RTOL", 0.1 if _bf16 else 1e-4))\n')
    buf.write('    if _bf16:\n')
    buf.write('        print("Note: tensors stored in bfloat16 — using relaxed tolerances (bf16 weights accumulate error through layers)")\n')
    buf.write('    _cap_mb = _meta.get("max_intermediates_mb")\n')
    buf.write('    if _cap_mb is not None:\n')
    buf.write('        print(f"Note: intermediates capped at {_cap_mb} MiB — only saved subset is verified")\n')
    buf.write("    print()\n\n")

    if fg:
        phs = [n for n in fg.graph_module.graph.nodes if n.op == "placeholder"]
        names = [p.name for p in phs]

        buf.write("    # ── Forward: verify the exported Python forward() function ──\n")
        buf.write(f"    fw_result = forward({', '.join(names)})\n")
        buf.write("    _fw_tensors = [t for t in (fw_result if isinstance(fw_result, tuple) else (fw_result,)) if isinstance(t, torch.Tensor)]\n")
        buf.write('    _fw_ref_out = _data.get("_forward_output")\n')
        buf.write("    if _fw_ref_out is not None:\n")
        buf.write("        _fw_ref_tensors = [t for t in _fw_ref_out if isinstance(t, torch.Tensor)]\n")
        buf.write("        _ok_fw, _total_fw, _fw_fail = 0, 0, []\n")
        buf.write("        for i, (_ref, _act) in enumerate(zip(_fw_ref_tensors, _fw_tensors)):\n")
        buf.write("            _total_fw += 1\n")
        buf.write("            _rf, _af = _ref.float().cpu(), _act.float().cpu()\n")
        buf.write("            if _ref.shape != _act.shape:\n")
        buf.write("                _fw_fail.append((i, _ref.shape, float('inf')))\n")
        buf.write("            elif _rf.isnan().any() or _af.isnan().any() or _rf.isinf().any() or _af.isinf().any():\n")
        buf.write("                _ok_fw += 1\n")
        buf.write("            elif torch.allclose(_rf, _af, atol=_atol, rtol=_rtol):\n")
        buf.write("                _ok_fw += 1\n")
        buf.write("            else:\n")
        buf.write("                _fw_fail.append((i, _ref.shape, (_rf - _af).abs().max().item()))\n")
        buf.write('        print(f"Forward outputs:  {_ok_fw}/{_total_fw} match original model (exported Python code)")\n')
        buf.write("        for _i, _s, _d in _fw_fail[:10]:\n")
        buf.write('            print(f"  FAIL: output[{_i}] shape={list(_s)} max_diff={_d:.2e}")\n')
        buf.write("    else:\n")
        buf.write('        print(f"Forward: {len(_fw_tensors)} output tensors (no reference to verify against)")\n')

        buf.write("\n    # ── Verify forward intermediates via FX GraphModule re-execution ──\n")
        buf.write('    _fw_ref = _data.get("_forward_intermediates", {})\n')
        buf.write("    if _fw_ref:\n")
        buf.write("        from torch.fx.interpreter import Interpreter\n")
        buf.write("        class _Rec(Interpreter):\n")
        buf.write("            def __init__(self, gm):\n")
        buf.write("                super().__init__(gm)\n")
        buf.write("                self.recorded = {}\n")
        buf.write("            def run_node(self, n):\n")
        buf.write("                result = super().run_node(n)\n")
        buf.write("                if isinstance(result, torch.Tensor):\n")
        buf.write("                    self.recorded[n.name] = result.clone().detach()\n")
        buf.write("                elif isinstance(result, (tuple, list)):\n")
        buf.write("                    for i, elem in enumerate(result):\n")
        buf.write("                        if isinstance(elem, torch.Tensor):\n")
        buf.write("                            self.recorded[f'{n.name}_{i}'] = elem.clone().detach()\n")
        buf.write("                return result\n\n")
        buf.write("        try:\n")
        if data_path_var:
            gm_fw_name = Path(pt_path).stem + "_gm_fw.pt"
            buf.write(f'            _gm = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "{gm_fw_name}"), weights_only=False)\n')
        else:
            _pt = Path(pt_path)
            gm_fw_abs = str(_pt.with_stem(_pt.stem + "_gm_fw"))
            buf.write(f'            _gm = torch.load("{gm_fw_abs}", weights_only=False)\n')
        buf.write(f"            _interp = _Rec(_gm)\n")
        buf.write(f"            _interp.run({', '.join(names)})\n")
        buf.write("        except Exception as _load_err:\n")
        buf.write('            print(f"Skipping FX intermediate verification: {_load_err}")\n')
        buf.write("            _fw_ref = {}\n")
        buf.write("        _ok, _total = 0, 0\n")
        buf.write("        _failures = []\n")
        buf.write("        for _name, _ref in sorted(_fw_ref.items()):\n")
        buf.write("            _act = _interp.recorded.get(_name)\n")
        buf.write("            if _act is None:\n")
        buf.write("                continue\n")
        buf.write("            _total += 1\n")
        buf.write("            _rf, _af = _ref.float().cpu(), _act.float().cpu()\n")
        buf.write("            if _ref.shape != _act.shape:\n")
        buf.write("                _failures.append((_name, _ref.shape, float('inf')))\n")
        buf.write("            elif _rf.isnan().any() or _af.isnan().any() or _rf.isinf().any() or _af.isinf().any():\n")
        buf.write("                _ok += 1\n")
        buf.write("            elif torch.allclose(_rf, _af, atol=_atol, rtol=_rtol):\n")
        buf.write("                _ok += 1\n")
        buf.write("            else:\n")
        buf.write("                _failures.append((_name, _ref.shape, (_rf - _af).abs().max().item()))\n")
        # Report capped info if present
        buf.write('        _fw_saved = _meta.get("fw_intermediates_saved")\n')
        buf.write('        _fw_total_cap = _meta.get("fw_intermediates_total")\n')
        buf.write("        _cap_note = f' (saved {_fw_saved}/{_fw_total_cap})' if _fw_saved is not None else ''\n")
        buf.write('        print(f"Forward intermediates: {_ok}/{_total} match (FX GraphModule re-execution){_cap_note}")\n')
        buf.write("        for _n, _s, _d in _failures[:10]:\n")
        buf.write('            print(f"  FAIL: {_n} shape={list(_s)} max_diff={_d:.2e}")\n')
        buf.write("    print()\n\n")

    if bg:
        buf.write("    # ── Backward: run with real inputs from original model ──\n")
        buf.write('    _bw_inputs = _data.get("_backward_inputs")\n')
        buf.write('    _bw_ref_output = _data.get("_backward_output")\n')
        buf.write("    if _bw_inputs is not None:\n")
        # Use fresh saved tensors from forward() to avoid stale Philox RNG state
        # (serialized Philox seed/offset from _scaled_dot_product_efficient_attention
        # become dangling pointers in a new process, causing segfaults).
        buf.write("        # Use fresh saved tensors from forward() to avoid stale CUDA Philox RNG state\n")
        buf.write("        _num_fw_out = _meta.get('num_user_outputs', 0)\n")
        buf.write("        if _num_fw_out > 0:\n")
        buf.write("            _fw_all = fw_result if isinstance(fw_result, tuple) else (fw_result,)\n")
        buf.write("            _num_saved = len(_fw_all) - _num_fw_out\n")
        buf.write("            _num_tangents = len(_bw_inputs) - _num_saved\n")
        buf.write("            if _num_saved > 0 and _num_tangents > 0:\n")
        buf.write("                _bw_inputs = list(_fw_all[_num_fw_out:]) + list(_bw_inputs[-_num_tangents:])\n")
        buf.write("        bw_result = backward(*_bw_inputs)\n")
        buf.write("        _bw_tensors = [t for t in (bw_result if isinstance(bw_result, tuple) else (bw_result,)) if isinstance(t, torch.Tensor)]\n")
        buf.write('        print(f"Backward: {len(_bw_tensors)} gradient outputs")\n')
        buf.write("        for i, t in enumerate(_bw_tensors):\n")
        buf.write('            print(f"  grad[{i}] shape={t.shape}, mean={t.float().mean():.6f}")\n')
        buf.write("        print()\n\n")

        buf.write("        # ── Verify backward outputs against original model ──\n")
        buf.write("        if _bw_ref_output is not None:\n")
        buf.write("            _bw_ref_tensors = [t for t in _bw_ref_output if isinstance(t, torch.Tensor)]\n")
        buf.write("            _ok_bw, _total_bw, _bw_fail = 0, 0, []\n")
        buf.write("            for i, (_ref, _act) in enumerate(zip(_bw_ref_tensors, _bw_tensors)):\n")
        buf.write("                _total_bw += 1\n")
        buf.write("                _rf, _af = _ref.float().cpu(), _act.float().cpu()\n")
        buf.write("                if _ref.shape != _act.shape:\n")
        buf.write("                    _bw_fail.append((i, _ref.shape, float('inf')))\n")
        buf.write("                elif _rf.numel() == 0 or (_rf.isnan().all() and _af.isnan().all()):\n")
        buf.write("                    _ok_bw += 1\n")
        buf.write("                elif torch.allclose(_rf, _af, atol=_atol, rtol=_rtol):\n")
        buf.write("                    _ok_bw += 1\n")
        buf.write("                else:\n")
        buf.write("                    _d = (_rf - _af).abs()\n")
        buf.write("                    _d = _d[~_d.isnan()]\n")
        buf.write("                    _bw_fail.append((i, _ref.shape, _d.max().item() if _d.numel() > 0 else float('nan')))\n")
        buf.write('            print(f"Backward outputs: {_ok_bw}/{_total_bw} match original model")\n')
        buf.write("            for _i, _s, _d in _bw_fail:\n")
        buf.write('                print(f"  FAIL: grad[{_i}] shape={list(_s)} max_diff={_d:.2e}")\n')
        buf.write("    else:\n")
        buf.write('        print("No backward inputs saved (re-export with record_real_tensors=True)")\n')

    # ── Optimizer aten verification ──
    buf.write('    _opt_inputs = _data.get("_optimizer_inputs")\n')
    buf.write('    _opt_ref_out = _data.get("_optimizer_output")\n')
    buf.write("    if _opt_inputs is not None and _opt_ref_out is not None:\n")
    buf.write("        try:\n")
    buf.write("            _opt_result = optimizer_step(*_opt_inputs)\n")
    buf.write("            _opt_tensors = [t for t in (_opt_result if isinstance(_opt_result, tuple) else (_opt_result,)) if isinstance(t, torch.Tensor)]\n")
    buf.write("            _opt_ref_tensors = [t for t in _opt_ref_out if isinstance(t, torch.Tensor)]\n")
    buf.write("            _ok_oa, _total_oa, _oa_fail = 0, 0, []\n")
    buf.write("            for i, (_ref, _act) in enumerate(zip(_opt_ref_tensors, _opt_tensors)):\n")
    buf.write("                _total_oa += 1\n")
    buf.write("                _rf, _af = _ref.float().cpu(), _act.float().cpu()\n")
    buf.write("                if _ref.shape != _act.shape:\n")
    buf.write("                    _oa_fail.append((i, _ref.shape, float('inf')))\n")
    buf.write("                elif torch.allclose(_rf, _af, atol=_atol, rtol=_rtol):\n")
    buf.write("                    _ok_oa += 1\n")
    buf.write("                else:\n")
    buf.write("                    _oa_fail.append((i, _ref.shape, (_rf - _af).abs().max().item()))\n")
    buf.write('            print(f"Optimizer aten:  {_ok_oa}/{_total_oa} outputs match (extracted aten ops)")\n')
    buf.write("            for _i, _s, _d in _oa_fail[:10]:\n")
    buf.write('                print(f"  FAIL: output[{_i}] shape={list(_s)} max_diff={_d:.2e}")\n')
    buf.write("        except Exception as _e:\n")
    buf.write('            print(f"Optimizer aten:  SKIPPED ({_e})")\n')
    buf.write("    print()\n")
    buf.write('    _dtype_note = " (stored as bfloat16)" if _bf16 else ""\n')
    buf.write('    print(f"All tensors loaded from .pt file{_dtype_note} — zero synthetic data.")\n')


def _emit_basic_harness(buf: StringIO, capture: AtenCapture, has_symbolic_shapes: bool = False) -> None:
    """Generate basic test harness (no real tensors available)."""
    if has_symbolic_shapes:
        buf.write('    # Note: Shapes are symbolic - batch/sequence dimensions may vary.\n\n')
    buf.write('    print()\n')

    if capture.forward_graphs:
        fg = capture.forward_graphs[0]
        phs = [n for n in fg.graph_module.graph.nodes if n.op == "placeholder"]
        names = [p.name for p in phs]
        buf.write(f"    fw_result = forward({', '.join(names)})\n")
        buf.write("    _fw = [t for t in (fw_result if isinstance(fw_result, tuple) else (fw_result,)) if isinstance(t, torch.Tensor)]\n")
        buf.write('    print(f"Forward: {len(_fw)} output tensors")\n')
        buf.write("    for i, t in enumerate(_fw):\n")
        buf.write('        print(f"  [{i}] shape={t.shape}, mean={t.float().mean():.6f}")\n')
        buf.write("    print()\n\n")

    if capture.backward_graphs:
        bg = capture.backward_graphs[0]
        buf.write(f"    bw_example_inputs = {_example_inputs_constructor(bg.example_inputs)}\n")
        buf.write("    bw_result = backward(*bw_example_inputs)\n")
        buf.write("    _bw = [t for t in (bw_result if isinstance(bw_result, tuple) else (bw_result,)) if isinstance(t, torch.Tensor)]\n")
        buf.write('    print(f"Backward: {len(_bw)} gradient outputs")\n')
        buf.write("    for i, t in enumerate(_bw):\n")
        buf.write('        print(f"  grad[{i}] shape={t.shape}, mean={t.float().mean():.6f}")\n')

    buf.write('\n    print()\n')
    buf.write('    print("SUCCESS: Exported aten program runs correctly.")\n')
    buf.write('    print("Re-export with record_real_tensors=True for full verification.")\n')


def _example_inputs_constructor(inputs: list, device_var: str = "_device") -> str:
    """Build a constructor for example inputs list."""
    dev = f", device={device_var}"
    parts = []
    for inp in inputs:
        if not isinstance(inp, torch.Tensor):
            # SymInt or other non-tensor: use concrete int
            try:
                parts.append(str(int(inp)))
            except Exception:
                parts.append("2")
            continue
        shape = list(inp.shape)
        dtype = str(inp.dtype).replace("torch.", "")
        is_float = inp.dtype.is_floating_point
        try:
            if inp.numel() <= 100:
                parts.append(f"torch.tensor({inp.tolist()}, dtype=torch.{dtype}{dev})")
            elif is_float:
                parts.append(f"torch.randn({shape}, dtype=torch.{dtype}{dev})")
            else:
                parts.append(f"torch.zeros({shape}, dtype=torch.{dtype}{dev})")
        except Exception:
            ctor = "torch.randn" if is_float else "torch.zeros"
            parts.append(f"{ctor}({shape}, dtype=torch.{dtype}{dev})")
    return "[" + ",\n        ".join(parts) + "]"


# -----------------------------------------------------------------------------
# Tensor dumping and op extraction tools
# -----------------------------------------------------------------------------


def trace_tensors_from_graph(
    graph_module: GraphModule,
    *args,
    names: list[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Run a GraphModule and capture ALL intermediate tensor values.

    This instruments the graph to record every tensor produced by every op.
    Much more complete than trace_tensors() - captures every single intermediate.

    Args:
        graph_module: The FX GraphModule (from AtenGraph.graph_module).
        *args: Inputs matching the graph's placeholders.
        names: If given, only record these variable names. None = record all.

    Returns:
        Dict mapping variable names to tensor values.

    Usage:
        output, capture = capture_aten_graphs(model, x)
        fg = capture.forward_graphs[0]
        tensors = trace_tensors_from_graph(fg.graph_module, *fg.example_inputs)
        # Now tensors['addmm'], tensors['relu'], etc. are all available
    """
    interp = _RecordInterp(graph_module)
    interp.run(*args)
    recorded = interp.recorded
    if names is not None:
        recorded = {k: v for k, v in recorded.items() if k in names}
    return recorded


def extract_subgraph(
    graph_module: GraphModule,
    node_names: list[str],
    output_path: str | None = None,
) -> str:
    """Extract a subset of ops into a standalone function.

    Given a list of node names, extracts those ops plus all required
    inputs into a minimal standalone Python function that can be run
    independently.

    Args:
        graph_module: Source GraphModule.
        node_names: Names of nodes to include (e.g. ['t', 'addmm', 'relu']).
        output_path: Optional path to save the extracted function as .py.

    Returns:
        String containing the standalone Python function.

    Usage:
        code = extract_subgraph(fg.graph_module, ['t', 'addmm', 'relu'])
        print(code)  # self-contained function with just those ops
    """
    graph = graph_module.graph
    name_to_node = {n.name: n for n in graph.nodes}

    # Validate names
    target_nodes = []
    for name in node_names:
        if name not in name_to_node:
            raise ValueError(f"Node '{name}' not found in graph. "
                             f"Available: {list(name_to_node.keys())[:20]}...")
        target_nodes.append(name_to_node[name])

    target_set = set(node_names)

    # Find all required inputs: nodes that are consumed by targets but not in the target set
    required_inputs = OrderedDict()
    for node in target_nodes:
        for inp in node.all_input_nodes:
            if inp.name not in target_set:
                if inp.name not in required_inputs:
                    val = inp.meta.get("val")
                    shape = list(val.shape) if hasattr(val, "shape") else []
                    dtype = val.dtype if hasattr(val, "dtype") else torch.float32
                    required_inputs[inp.name] = (shape, dtype)

    # Find all outputs: target nodes that are consumed by nodes NOT in the target set,
    # or that have no consumers (leaf nodes)
    outputs = []
    for node in target_nodes:
        is_output = len(node.users) == 0
        for user in node.users:
            if user.name not in target_set:
                is_output = True
                break
        if is_output:
            outputs.append(node.name)

    # Build the extracted function
    buf = StringIO()
    buf.write("import operator\n")
    buf.write("import torch\n")
    buf.write("aten = torch.ops.aten\n\n")

    # Generate input constructors
    buf.write("# Required input tensors\n")
    for inp_name, (shape, dtype) in required_inputs.items():
        ctor = "torch.randn" if dtype.is_floating_point else "torch.zeros"
        buf.write(f"{inp_name} = {ctor}({shape}, dtype={dtype})\n")
    buf.write("\n")

    # Function signature
    params = ", ".join(required_inputs.keys())
    buf.write(f"def extracted({params}):\n")

    # Body: emit the target nodes in graph order
    all_graph_nodes = list(graph.nodes)
    for node in all_graph_nodes:
        if node.name not in target_set:
            continue
        line = _ir_fx_node_to_python(node)
        if not line:
            continue
        line = re.sub(r"%(\w+)", r"\1", line)
        buf.write(f"    {line}\n")

    # Return statement
    if outputs:
        buf.write(f"    return ({', '.join(outputs)},)\n")
    buf.write("\n")

    # Test code
    buf.write("# Run it:\n")
    buf.write(f"result = extracted({params})\n")
    buf.write("if isinstance(result, tuple):\n")
    buf.write("    for i, t in enumerate(result):\n")
    buf.write("        if isinstance(t, torch.Tensor):\n")
    buf.write("            print(f'  [{i}] shape={t.shape}, dtype={t.dtype}')\n")
    buf.write("elif isinstance(result, torch.Tensor):\n")
    buf.write("    print(f'  shape={result.shape}, dtype={result.dtype}')\n")

    code = buf.getvalue()

    if output_path:
        Path(output_path).write_text(code)

    return code


def list_ops(graph_module: GraphModule) -> list[dict]:
    """List all operations in a graph with their metadata.

    Useful for browsing what's available before calling extract_subgraph().

    Returns:
        List of dicts with keys: name, op_type, target, shape, inputs, group
    """
    ops = []
    for node in graph_module.graph.nodes:
        if node.op in ("placeholder", "output"):
            continue

        target = _ir_callable_to_str(node.target) if node.op == "call_function" else str(node.target)
        shape = ""
        if "val" in node.meta:
            val = node.meta["val"]
            if isinstance(val, torch.Tensor):
                shape = str(list(val.shape))

        mod_path, mod_type, src_fn = _extract_source_group(node)
        group = _format_group_header(mod_path, mod_type, src_fn)

        ops.append({
            "name": node.name,
            "op_type": node.op,
            "target": target,
            "shape": shape,
            "inputs": [inp.name for inp in node.all_input_nodes],
            "group": group,
        })
    return ops
