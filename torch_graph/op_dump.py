"""Per-operation-group tensor dumping.

Dumps tensors grouped by "sets of operations" — where a single line of Python
source code (e.g., `x = F.relu(self.fc1(x))`) decomposes into multiple aten ops
(t, addmm, relu), each touching several tensors.

Produces HDF5 or PT files where:
  - All tensors touched by a group of ops live in one file
  - Within the file, you can see inputs and outputs per individual operation
  - Outputs = tensors produced by the op (not pre-existing)
  - Aliasing: if op A produces tensor T and op B consumes T, T appears under
    both opA/outputs/T and opB/inputs/T without data duplication (HDF5 hard links)
  - Rich metadata on every tensor, op, and group (shape, dtype, source code, module)

Supports subsetting (by node names, pattern, module path, op category, source
lines) and multiple grouping strategies (line, module, op, all).
"""

from __future__ import annotations

import fnmatch
import logging
import numpy as np
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
from torch.fx import GraphModule, Node

from torch_graph._utils import (
    RecordingInterpreter as _RecordInterp,
    clean_self_path as _clean_self_path,
    is_fake as _is_fake,
    materialize_tensor as _materialize_tensor,
    short_name as _short_name,
)
logger = logging.getLogger(__name__)

from torch_graph.export import (
    AtenCapture,
    _extract_source_group,
    _group_key,
    _build_primal_map,
    capture_aten_graphs,
)


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

_DTYPE_SHORT = {
    "torch.float32": "fp32", "torch.float16": "fp16", "torch.bfloat16": "bf16",
    "torch.float64": "fp64", "torch.int64": "i64", "torch.int32": "i32",
    "torch.int16": "i16", "torch.int8": "i8", "torch.uint8": "u8",
    "torch.bool": "bool",
    "torch.complex64": "c64", "torch.complex128": "c128",
    "torch.float8_e4m3fn": "fp8e4m3", "torch.float8_e5m2": "fp8e5m2",
    "torch.float8_e4m3fnuz": "fp8e4m3fnuz", "torch.float8_e5m2fnuz": "fp8e5m2fnuz",
    "torch.uint16": "u16", "torch.uint32": "u32", "torch.uint64": "u64",
}


# -----------------------------------------------------------------------------
# Data model
# -----------------------------------------------------------------------------

@dataclass
class OpInfo:
    """Per-operation input/output breakdown within a group."""
    name: str
    target: str
    input_names: list[str] = field(default_factory=list)
    output_names: list[str] = field(default_factory=list)
    replay_script: str = ""
    source_file: str = ""
    source_line: int = 0
    source_code: str = ""
    module_path: str = ""
    module_type: str = ""


@dataclass
class OpGroup:
    """A group of aten operations with their collective inputs/outputs."""
    name: str
    node_names: list[str] = field(default_factory=list)
    all_node_names: list[str] = field(default_factory=list)
    external_inputs: list[str] = field(default_factory=list)
    external_outputs: list[str] = field(default_factory=list)
    ops: list[OpInfo] = field(default_factory=list)
    replay_script: str = ""
    # Metadata
    module_path: str = ""
    module_type: str = ""
    source_file: str = ""
    source_line: int = 0
    source_code: str = ""
    # Triton kernel (populated when group_by="triton")
    kernel_code: str = ""
    kernel_type: str = ""


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _short_dtype(dtype_str: str) -> str:
    """Convert torch dtype string to short form, e.g. 'torch.float32' -> 'fp32'."""
    return _DTYPE_SHORT.get(dtype_str, dtype_str.replace("torch.", ""))



def _sanitize_h5_name(name: str) -> str:
    """Sanitize for HDF5 group/dataset names (no '/' or null bytes)."""
    return name.replace("/", "|").replace("\x00", "")


def _get_node_target_str(node: Node) -> str:
    """Get a normalized string representation of a node's target."""
    if hasattr(node.target, "name") and callable(node.target.name):
        return node.target.name()
    if hasattr(node.target, "__name__"):
        return node.target.__name__
    return str(node.target)


# Targets considered "plumbing" — they just reshape/index/alias without computing
_VIEW_TARGETS = {
    "view", "reshape", "permute", "transpose", "expand", "contiguous",
    "unsqueeze", "squeeze", "slice", "select", "split", "unbind", "chunk",
    "narrow", "as_strided", "alias", "detach", "clone", "t", "flatten",
    "getitem",  # tuple indexing after split/unbind
}


def _is_view_op(node: Node) -> bool:
    """Return True if this node is a view/reshape/index plumbing op."""
    if node.op not in ("call_function", "call_method", "call_module"):
        return False
    target = _get_node_target_str(node).lower()
    # Strip namespace: aten::view.default -> view
    if "::" in target:
        target = target.split("::")[-1]
    target = target.split(".")[0]
    return target in _VIEW_TARGETS


def _extract_source_info_for_node(node: Node, source_map: dict | None = None) -> dict:
    """Extract source file/line/code from a node's metadata."""
    info: dict[str, Any] = {}

    # Parse stack_trace
    trace = node.meta.get("stack_trace", "")
    if trace:
        lines = trace.strip().split("\n")
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i]
            if "torch" in line.lower() and "test_repo" not in line.lower():
                continue
            m = re.match(r'\s*File "([^"]+)", line (\d+), in (\w+)', line)
            if m:
                info["file"] = m.group(1)
                info["line"] = int(m.group(2))
                if i + 1 < len(lines):
                    code = lines[i + 1].strip()
                    if code and not code.startswith("File "):
                        info["code"] = code
                break

    # Fallback: source_map
    if "file" not in info and source_map:
        src_fn = node.meta.get("source_fn_stack") or node.meta.get("fwd_source_fn_stack", [])
        src_fn_key = ""
        if src_fn:
            for item in src_fn:
                if isinstance(item, tuple) and len(item) >= 1:
                    src_fn_key = str(item[0])
        if src_fn_key and src_fn_key in source_map:
            st = source_map[src_fn_key]
            if st.file:
                info["file"] = st.file
            if st.line:
                info["line"] = st.line
            if st.code:
                info["code"] = st.code

    return info


def _get_op_category(node: Node) -> str | None:
    """Categorize a node into an op category. Returns None if not a compute op."""
    if node.op not in ("call_function", "call_method", "call_module"):
        return None

    name = _get_node_target_str(node).lower()

    activation_kw = {"relu", "gelu", "silu", "sigmoid", "tanh", "softmax", "dropout"}
    norm_kw = {"layer_norm", "batch_norm", "group_norm", "norm"}
    linalg_kw = {"mm", "bmm", "matmul", "linear", "addmm", "conv"}
    reduction_kw = {"sum", "mean", "max", "min", "amax", "var", "std"}
    view_kw = {"view", "reshape", "permute", "transpose", "expand", "contiguous",
               "unsqueeze", "squeeze", "slice", "select"}
    attention_kw = {"attention", "scaled_dot_product"}
    arith_kw = {"add", "sub", "mul", "div", "pow", "neg", "exp", "log", "sqrt", "rsqrt"}

    for keywords, cat in [
        (attention_kw, "attention"),
        (activation_kw, "activation"),
        (norm_kw, "normalization"),
        (linalg_kw, "linear_algebra"),
        (reduction_kw, "reduction"),
        (view_kw, "view_reshape"),
        (arith_kw, "arithmetic"),
    ]:
        if any(kw in name for kw in keywords):
            return cat
    return "other"


# -----------------------------------------------------------------------------
# Tensor metadata
# -----------------------------------------------------------------------------

def _build_tensor_meta(
    gm: GraphModule,
    source_map: dict | None = None,
    capture: AtenCapture | None = None,
) -> dict[str, dict]:
    """Build rich metadata for every node in the graph.

    Returns dict mapping node_name -> {shape, dtype, human_name, source_file,
    source_line, source_code, module_path, module_type, target, category, ...}
    """
    primal_map = _build_primal_map(gm, capture) if capture else {}
    meta: dict[str, dict] = {}

    for node in gm.graph.nodes:
        info: dict[str, Any] = {"name": node.name, "op": node.op}

        # Shape, dtype, strides, contiguity, device from meta["val"]
        val = node.meta.get("val")
        if val is not None:
            t = val
            if isinstance(val, (tuple, list)):
                t = next((v for v in val if isinstance(v, torch.Tensor)), None)
            if isinstance(t, torch.Tensor):
                info["shape"] = list(t.shape)
                info["dtype"] = str(t.dtype)
                info["device"] = str(t.device)
                if t.dim() > 0:
                    info["strides"] = list(t.stride())
                    info["is_contiguous"] = t.is_contiguous()
                else:
                    info["strides"] = []
                    info["is_contiguous"] = True

        # Human name from primal_map
        if node.name in primal_map:
            info["human_name"] = primal_map[node.name]

        # Source info
        si = _extract_source_info_for_node(node, source_map)
        if "file" in si:
            info["source_file"] = si["file"]
        if "line" in si:
            info["source_line"] = si["line"]
        if "code" in si:
            info["source_code"] = si["code"]

        # Module info
        mod_path, mod_type, src_fn = _extract_source_group(node)
        if mod_path:
            info["module_path"] = _clean_self_path(mod_path)
        if mod_type:
            info["module_type"] = mod_type

        # Op target and category
        if node.op in ("call_function", "call_method", "call_module"):
            info["target"] = _get_node_target_str(node)
            cat = _get_op_category(node)
            if cat:
                info["category"] = cat

        meta[node.name] = info

    return meta


# -----------------------------------------------------------------------------
# Human-readable tensor aliases from source code
# -----------------------------------------------------------------------------

def _parse_assignment_targets(code: str) -> list[str] | None:
    """Parse the LHS of an assignment from a source code line.

    Returns None if not a recognizable assignment, ["x"] for ``x = expr``,
    or ["q", "k", "v"] for ``q, k, v = expr``.
    """
    import ast
    code = code.strip()
    if "=" not in code or code.startswith("return "):
        return None
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    if not tree.body or not isinstance(tree.body[0], ast.Assign):
        return None
    assign = tree.body[0]
    if len(assign.targets) != 1:
        return None
    target = assign.targets[0]
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        names = []
        for elt in target.elts:
            if isinstance(elt, ast.Name):
                names.append(elt.id)
            else:
                return None
        return names if names else None
    return None


def _build_human_aliases(
    gm: GraphModule, source_map: dict | None = None,
) -> dict[str, str]:
    """Infer human-readable names for intermediate tensors from source code.

    Handles:
      - Tuple unpacking:  ``q, k, v = qkv.split(C, dim=2)``
        → getitem(split, 0) → "q", getitem(split, 1) → "k", ...
      - Simple assignment: ``x = self.ln_f(x)``
        → the node consumed by the *next* source line → "x"

    Returns dict mapping FX node name → human-readable alias.
    """
    from collections import defaultdict

    aliases: dict[str, str] = {}

    # Group compute nodes by source line
    line_nodes: dict[tuple[str, int], list[Node]] = defaultdict(list)
    line_code: dict[tuple[str, int], str] = {}
    node_line: dict[str, tuple[str, int]] = {}

    for node in gm.graph.nodes:
        if node.op not in ("call_function", "call_method", "call_module"):
            continue
        si = _extract_source_info_for_node(node, source_map)
        f, ln, code = si.get("file", ""), si.get("line", 0), si.get("code", "")
        if f and ln:
            key = (f, ln)
            line_nodes[key].append(node)
            if key not in line_code and code:
                line_code[key] = code
            node_line[node.name] = key

    for key, nodes in line_nodes.items():
        code = line_code.get(key, "")
        if not code:
            continue
        targets = _parse_assignment_targets(code)
        if targets is None:
            continue

        if len(targets) > 1:
            # Tuple unpacking — group getitems by their parent op
            parent_groups: dict[str, list[tuple[int, Node]]] = defaultdict(list)
            for n in nodes:
                ts = _get_node_target_str(n).lower()
                if "getitem" in ts and len(n.args) >= 2 and isinstance(n.args[1], int):
                    parent = n.args[0]
                    if hasattr(parent, "name"):
                        parent_groups[parent.name].append((n.args[1], n))

            for _parent, items in parent_groups.items():
                items.sort(key=lambda x: x[0])
                indices = [i for i, _ in items]
                if indices == list(range(len(targets))):
                    for (idx, node), name in zip(items, targets):
                        aliases[node.name] = name
        else:
            # Simple assignment — find nodes consumed by a different source line
            name = targets[0]
            external = []
            for n in nodes:
                consumed_externally = False
                for user in n.users:
                    ukey = node_line.get(user.name)
                    if ukey is None or ukey != key:
                        consumed_externally = True
                        break
                if consumed_externally or not n.users:
                    external.append(n)

            if not external:
                continue

            # Prefer non-getitem nodes; fall back to getitem(0) for tuple-returning ops
            best = []
            for n in external:
                ts = _get_node_target_str(n).lower()
                if "getitem" not in ts:
                    best.append(n)
            if not best:
                for n in external:
                    if len(n.args) >= 2 and isinstance(n.args[1], int) and n.args[1] == 0:
                        best.append(n)
            if not best:
                best = external

            for n in best:
                aliases[n.name] = name

    return aliases


# -----------------------------------------------------------------------------
# Node selection (subsetting)
# -----------------------------------------------------------------------------

def select_nodes(
    gm: GraphModule,
    *,
    nodes: list[str] | None = None,
    pattern: str | list[str] | None = None,
    module: str | None = None,
    category: str | None = None,
    lines: str | None = None,
    source_map: dict | None = None,
) -> set[str]:
    """Select a subset of compute nodes by composable selectors (intersection).

    All selectors are optional — when multiple are given, the result is their
    intersection. If none are given, returns all compute nodes.
    """
    compute_nodes = [
        n for n in gm.graph.nodes
        if n.op in ("call_function", "call_method", "call_module")
    ]

    # Start with all compute nodes, intersect each selector
    sets: list[set[str]] = []

    if nodes is not None:
        sets.append(set(nodes))

    if pattern is not None:
        patterns = [pattern] if isinstance(pattern, str) else pattern
        matched = set()
        for n in compute_nodes:
            if any(fnmatch.fnmatch(n.name, p) for p in patterns):
                matched.add(n.name)
        sets.append(matched)

    if module is not None:
        matched = set()
        for n in compute_nodes:
            mod_path, _, _ = _extract_source_group(n)
            clean = _clean_self_path(mod_path)
            if module in clean:
                matched.add(n.name)
        sets.append(matched)

    if category is not None:
        matched = set()
        for n in compute_nodes:
            cat = _get_op_category(n)
            if cat == category:
                matched.add(n.name)
        sets.append(matched)

    if lines is not None:
        matched = set()
        # Parse "file.py:40-50,55,100-120" — comma-separated line specs.
        # Each spec is either a single line "N" or a range "N-M".
        # The file prefix is everything before the last ":" that precedes digits.
        m = re.match(r"(.+):(.+)$", lines)
        if m:
            file_pattern = m.group(1)
            specs = m.group(2)
            line_set: set[int] = set()
            for spec in specs.split(","):
                spec = spec.strip()
                rm = re.match(r"(\d+)-(\d+)$", spec)
                if rm:
                    for ln in range(int(rm.group(1)), int(rm.group(2)) + 1):
                        line_set.add(ln)
                elif spec.isdigit():
                    line_set.add(int(spec))
            for n in compute_nodes:
                si = _extract_source_info_for_node(n, source_map)
                sf = si.get("file", "")
                sl = si.get("line", 0)
                if sf and sl:
                    short_file = sf.rsplit("/", 1)[-1]
                    if (file_pattern in sf or file_pattern in short_file) and sl in line_set:
                        matched.add(n.name)
        sets.append(matched)

    if not sets:
        return {n.name for n in compute_nodes}

    result = sets[0]
    for s in sets[1:]:
        result &= s
    return result


# -----------------------------------------------------------------------------
# Dependency closure
# -----------------------------------------------------------------------------

def expand_closure(gm: GraphModule, node_names: set[str]) -> set[str]:
    """Expand a set of node names to include all upstream dependencies.

    Walks backward from each selected node and includes all compute-op
    ancestors. Placeholders are not included (they're handled as external inputs).
    """
    name_to_node = {n.name: n for n in gm.graph.nodes}
    expanded = set(node_names)
    visited: set[str] = set()

    def _walk(n: Node):
        if n.name in visited:
            return
        visited.add(n.name)
        if n.op in ("call_function", "call_method", "call_module"):
            expanded.add(n.name)
        for inp in n.all_input_nodes:
            _walk(inp)

    for name in list(node_names):
        if name in name_to_node:
            _walk(name_to_node[name])

    return expanded


# -----------------------------------------------------------------------------
# Replay script generation
# -----------------------------------------------------------------------------

def _arg_to_python(arg, produced: set[str]) -> str:
    """Convert an FX node argument to a Python expression for the replay script."""
    if isinstance(arg, Node):
        name = arg.name
        if name in produced:
            return f'outputs["{name}"]'
        return f'inputs["{name}"]'
    if isinstance(arg, (list, tuple)):
        inner = ", ".join(_arg_to_python(a, produced) for a in arg)
        if isinstance(arg, list):
            return f"[{inner}]"
        return f"({inner},)" if len(arg) == 1 else f"({inner})"
    if isinstance(arg, torch.dtype):
        return str(arg)
    if isinstance(arg, torch.device):
        return f'torch.device("{arg}")'
    if isinstance(arg, torch.memory_format):
        return str(arg)
    if isinstance(arg, torch.layout):
        return str(arg)
    if arg is None or isinstance(arg, (bool, int, float, str)):
        return repr(arg)
    return repr(arg)


def _target_to_python(node: Node) -> str:
    """Convert a node's target to a callable Python expression."""
    target = node.target
    s = str(target)
    if s.startswith("aten."):
        return f"torch.ops.{s}"
    if hasattr(target, "__module__") and target.__module__ == "_operator":
        return f"operator.{target.__name__}"
    if hasattr(target, "__module__") and hasattr(target, "__qualname__"):
        mod = target.__module__ or ""
        if mod.startswith("torch"):
            return f"{mod}.{target.__qualname__}"
    if hasattr(target, "__name__"):
        name = target.__name__
        if name == "getitem":
            return "operator.getitem"
        return name
    return s


def _build_op_replay(node: Node, produced: set[str]) -> str:
    """Build a single-line replay for one FX node."""
    call = _target_to_python(node)
    parts = [_arg_to_python(a, produced) for a in node.args]
    for k, v in node.kwargs.items():
        parts.append(f"{k}={_arg_to_python(v, produced)}")
    return f'outputs["{node.name}"] = {call}({", ".join(parts)})'


def _build_group_replay(gm: GraphModule, all_node_names: list[str]) -> str:
    """Build a replay script for a group of FX nodes.

    The script assumes `inputs` dict has all external inputs as torch tensors,
    and `outputs` dict is initialized empty. All intermediate and final results
    are placed in `outputs`.
    """
    name_to_node = {n.name: n for n in gm.graph.nodes}
    produced: set[str] = set()

    lines: list[str] = []

    for nname in all_node_names:
        node = name_to_node.get(nname)
        if node is None or node.op not in ("call_function", "call_method", "call_module"):
            continue
        lines.append(_build_op_replay(node, produced))
        produced.add(nname)

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Triton per-invocation mapping
# -----------------------------------------------------------------------------

def _build_triton_call_map_for_graph(gm: GraphModule, tc) -> dict[str, str]:
    """Build a per-invocation mapping from FX node name to unique call key.

    Unlike ``build_kernel_node_map`` (which maps node → kernel *name*),
    this returns node → ``"kernel_name#invocation_idx"`` so that each
    call of the same kernel becomes its own group.
    """
    from torch_graph.triton import _extract_short_op

    kernel_defs = {k.name: k for k in tc.kernels}

    op_queues: dict[str, list[str]] = {}
    all_compute: list[str] = []
    node_map: dict[str, Node] = {}
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        node_map[node.name] = node
        short_op = _extract_short_op(str(node.target))
        if short_op:
            op_queues.setdefault(short_op, []).append(node.name)
        all_compute.append(node.name)

    propagate_only = {
        "view", "t", "transpose", "reshape", "permute", "expand",
        "contiguous", "unsqueeze", "squeeze", "getitem",
        "clone", "detach", "split",
    }

    node_to_call: dict[str, str] = {}
    call_count: dict[str, int] = {}

    for call in tc.call_sequence:
        kname = call["name"]
        idx = call_count.get(kname, 0)
        call_count[kname] = idx + 1
        call_key = f"{kname}#{idx}"

        if call["type"] == "extern":
            fn = call["fn"]
            if fn in op_queues and op_queues[fn]:
                node_to_call[op_queues[fn].pop(0)] = call_key
        else:
            kdef = kernel_defs.get(kname)
            if not kdef:
                continue
            for aten_op in kdef.fused_aten_ops:
                short_op = aten_op.replace("aten.", "").replace("prims.", "")
                if short_op in propagate_only:
                    continue
                if short_op in op_queues and op_queues[short_op]:
                    node_to_call[op_queues[short_op].pop(0)] = call_key

    native_ops = {
        "_scaled_dot_product_efficient_attention": "cuda_sdpa",
        "_scaled_dot_product_flash_attention": "cuda_sdpa",
    }
    for node_name in all_compute:
        if node_name in node_to_call:
            continue
        node = node_map.get(node_name)
        if not node:
            continue
        short_op = _extract_short_op(str(node.target))
        group = native_ops.get(short_op) if short_op else None
        if group:
            idx = call_count.get(group, 0)
            call_count[group] = idx + 1
            node_to_call[node_name] = f"{group}#{idx}"

    for node_name in all_compute:
        if node_name in node_to_call:
            continue
        node = node_map.get(node_name)
        if not node:
            continue
        for arg in node.args:
            if hasattr(arg, "name") and arg.name in node_to_call:
                node_to_call[node_name] = node_to_call[arg.name]
                break

    for node_name in reversed(all_compute):
        if node_name in node_to_call:
            continue
        node = node_map.get(node_name)
        if not node:
            continue
        for user in node.users:
            if hasattr(user, "name") and user.name in node_to_call:
                node_to_call[node_name] = node_to_call[user.name]
                break

    return node_to_call


def _build_triton_call_map(capture: AtenCapture) -> dict[str, str]:
    """Forward-graph per-invocation call map (convenience wrapper)."""
    tc = capture.triton_capture
    if tc is None or not capture.forward_graphs:
        return {}
    return _build_triton_call_map_for_graph(
        capture.forward_graphs[0].graph_module, tc)


def _build_backward_triton_call_map(capture: AtenCapture) -> dict[str, str]:
    """Backward-graph per-invocation call map."""
    tc = getattr(capture, "backward_triton_capture", None)
    if tc is None or not capture.backward_graphs:
        return {}
    return _build_triton_call_map_for_graph(
        capture.backward_graphs[0].graph_module, tc)


# -----------------------------------------------------------------------------
# Op-group analysis
# -----------------------------------------------------------------------------

def build_op_groups(
    gm: GraphModule,
    node_names: set[str],
    group_by: str = "line",
    source_map: dict | None = None,
    hide_views: bool = False,
    kernel_map: dict[str, str] | None = None,
) -> list[OpGroup]:
    """Build OpGroups from selected nodes using the chosen grouping strategy.

    When hide_views=True, view/reshape/getitem ops are omitted from per-op
    breakdown and group I/O is recomputed to "see through" the plumbing.
    """
    selected_set = set(node_names)

    # When hide_views + group_by="op", skip view nodes entirely from grouping
    # (each would become its own empty group). For multi-node groups (line/module/all),
    # view nodes are filtered inside the group instead.
    name_to_node_all = {n.name: n for n in gm.graph.nodes}
    if hide_views and group_by == "op":
        selected_set = {
            name for name in selected_set
            if name in name_to_node_all and not _is_view_op(name_to_node_all[name])
        }

    # Group nodes by the chosen key, also track source code per key
    groups_map: OrderedDict[str, list[Node]] = OrderedDict()
    key_code: dict[str, str] = {}

    for node in gm.graph.nodes:
        if node.name not in selected_set:
            continue

        si = _extract_source_info_for_node(node, source_map)
        code = si.get("code", "")

        if group_by == "line":
            sf = si.get("file", "")
            sl = si.get("line", 0)
            if sf and sl:
                short_file = sf.rsplit("/", 1)[-1]
                # Disambiguate same source line across different module instances
                # (e.g., blocks.0 vs blocks.1 both hit model.py:48)
                mod_path, _, _ = _extract_source_group(node)
                mod_prefix = _clean_self_path(mod_path, keep_self=False)
                # Use top-level distinguishing prefix (e.g., "blocks.0" from "blocks.0.ln_1")
                # by finding the part with a numeric index
                disambig = ""
                if mod_prefix:
                    parts = mod_prefix.split(".")
                    # Accumulate up to and including the first indexed component
                    acc = []
                    for p in parts:
                        acc.append(p)
                        if p.isdigit():
                            break
                    disambig = ".".join(acc)
                if disambig:
                    key = f"{short_file}:{sl}@{disambig}"
                else:
                    key = f"{short_file}:{sl}"
            else:
                key = _group_key(node) or node.name
        elif group_by == "module":
            mod_path, mod_type, _ = _extract_source_group(node)
            clean = _clean_self_path(mod_path)
            key = clean if clean else node.name
        elif group_by == "op":
            key = node.name
        elif group_by == "all":
            key = "all"
        elif group_by == "triton":
            if kernel_map and node.name in kernel_map:
                key = kernel_map[node.name]
            else:
                key = "_unmapped"
        else:
            raise ValueError(f"Unknown group_by: {group_by!r}. Use 'line', 'module', 'op', 'triton', or 'all'.")

        groups_map.setdefault(key, []).append(node)
        if key not in key_code and code:
            key_code[key] = code

    # Build OpGroup for each group
    result = []
    display_idx = 0
    for key, group_nodes in groups_map.items():
        all_group_node_names = [n.name for n in group_nodes]
        group_node_names = list(all_group_node_names)
        group_set = set(group_node_names)

        # Per-op breakdown with rich metadata
        ops = []
        for n in group_nodes:
            target_str = _get_node_target_str(n)
            op_inputs = [inp.name for inp in n.all_input_nodes]
            op_outputs = [n.name]
            op_si = _extract_source_info_for_node(n, source_map)
            mod_path, mod_type, _ = _extract_source_group(n)
            clean_mod = _clean_self_path(mod_path)
            ops.append(OpInfo(
                name=n.name,
                target=target_str,
                input_names=op_inputs,
                output_names=op_outputs,
                source_file=op_si.get("file", ""),
                source_line=op_si.get("line", 0),
                source_code=op_si.get("code", ""),
                module_path=clean_mod,
                module_type=mod_type,
            ))

        # When hide_views is on, filter ops and recompute I/O to see through views
        if hide_views:
            name_to_node_local = {n.name: n for n in group_nodes}
            hidden = {n.name for n in group_nodes if _is_view_op(n)}
            ops = [op for op in ops if op.name not in hidden]
            # Rewrite remaining ops' input lists: if an input was produced by
            # a hidden op, trace back through the chain to its real source
            def _trace_through_hidden(node_name):
                """Follow hidden ops backward to find the real source."""
                visited = set()
                stack = [node_name]
                sources = []
                while stack:
                    cur = stack.pop()
                    if cur in visited:
                        continue
                    visited.add(cur)
                    if cur not in hidden:
                        sources.append(cur)
                    elif cur in name_to_node_local:
                        for inp in name_to_node_local[cur].all_input_nodes:
                            stack.append(inp.name)
                    else:
                        sources.append(cur)
                return sources

            for op in ops:
                new_inputs = []
                for iname in op.input_names:
                    for src in _trace_through_hidden(iname):
                        if src not in new_inputs:
                            new_inputs.append(src)
                op.input_names = new_inputs

            # Recompute visible set and external I/O
            visible_nodes = [n for n in group_nodes if n.name not in hidden]
            visible_set = {n.name for n in visible_nodes}

            ext_inputs = OrderedDict()
            for n in visible_nodes:
                # Use the rewritten op inputs
                matching_op = next((o for o in ops if o.name == n.name), None)
                if matching_op:
                    for iname in matching_op.input_names:
                        if iname not in visible_set and iname not in ext_inputs:
                            ext_inputs[iname] = True

            ext_outputs = OrderedDict()
            for n in group_nodes:
                if n.name in hidden:
                    continue
                is_ext = len(n.users) == 0
                for user in n.users:
                    if user.name not in group_set:
                        is_ext = True
                        break
                    # Also count as external if the only in-group consumers are hidden
                    if user.name in hidden:
                        # Check if hidden op's output eventually leaves the group
                        def _reaches_outside(uname, seen=None):
                            if seen is None:
                                seen = set()
                            if uname in seen:
                                return False
                            seen.add(uname)
                            if uname not in name_to_node_local:
                                return True
                            u = name_to_node_local[uname]
                            if not u.users:
                                return True
                            for uu in u.users:
                                if uu.name not in group_set:
                                    return True
                                if uu.name in hidden:
                                    if _reaches_outside(uu.name, seen):
                                        return True
                            return False
                        if _reaches_outside(user.name):
                            is_ext = True
                            break
                if is_ext:
                    ext_outputs[n.name] = True

            group_node_names = [n.name for n in visible_nodes]
        else:
            # External inputs: consumed by group but produced outside it
            ext_inputs = OrderedDict()
            for n in group_nodes:
                for inp in n.all_input_nodes:
                    if inp.name not in group_set and inp.name not in ext_inputs:
                        ext_inputs[inp.name] = True

            # External outputs: produced by group and consumed outside it (or leaf)
            ext_outputs = OrderedDict()
            for n in group_nodes:
                is_ext = len(n.users) == 0
                for user in n.users:
                    if user.name not in group_set:
                        is_ext = True
                        break
                if is_ext:
                    ext_outputs[n.name] = True

        # Skip empty groups (all ops hidden, no external I/O)
        if hide_views and not ops and not ext_inputs and not ext_outputs:
            continue

        # Extract metadata from first non-view node if possible
        first = group_nodes[0]
        if hide_views:
            for n in group_nodes:
                if not _is_view_op(n):
                    first = n
                    break
        mod_path, mod_type, _ = _extract_source_group(first)
        clean_mod = _clean_self_path(mod_path)
        si_first = _extract_source_info_for_node(first, source_map)
        source_code = key_code.get(key, si_first.get("code", ""))

        # Build descriptive group name
        group_display = _build_group_display_name(display_idx, key, group_by, source_code,
                                                   clean_mod, mod_type, first)

        result.append(OpGroup(
            name=group_display,
            node_names=group_node_names,
            all_node_names=all_group_node_names,
            external_inputs=list(ext_inputs.keys()),
            external_outputs=list(ext_outputs.keys()),
            ops=ops,
            module_path=clean_mod,
            module_type=mod_type,
            source_file=si_first.get("file", ""),
            source_line=si_first.get("line", 0),
            source_code=source_code,
        ))
        display_idx += 1

    return result


def _build_group_display_name(
    idx: int, key: str, group_by: str, source_code: str,
    module_path: str, module_type: str, first_node: Node,
) -> str:
    """Build a descriptive group name that shows actual code/context."""
    if group_by == "line":
        # Format: 003_model.py:22 [blocks.0] | qkv = self.c_attn(x)
        display_key = key
        disambig_part = ""
        if "@" in key:
            base_key, disambig = key.split("@", 1)
            display_key = base_key
            disambig_part = f" [{disambig}]"
        if source_code:
            code = source_code.strip()
            if len(code) > 200:
                code = code[:197] + "..."
            return f"{idx:03d}_{display_key}{disambig_part} | {code}"
        return f"{idx:03d}_{display_key}{disambig_part}"
    elif group_by == "module":
        type_suffix = f" ({module_type})" if module_type else ""
        return f"{idx:03d}_{module_path}{type_suffix}" if module_path else f"{idx:03d}_{key}"
    elif group_by == "op":
        return f"{idx:03d}_{first_node.name}"
    elif group_by == "all":
        return f"{idx:03d}_all"
    elif group_by == "triton":
        display = key.rsplit("#", 1)[0] if "#" in key else key
        return f"{idx:03d}_{display}"
    return f"{idx:03d}_{key}"


# -----------------------------------------------------------------------------
# Tensor collection
# -----------------------------------------------------------------------------

def _collect_inputs_only(
    gm: GraphModule,
    capture: AtenCapture | None,
    kind: str = "forward",
) -> dict[str, torch.Tensor]:
    """Collect only placeholder tensors + the first forward output for minimal H5.

    In AOT autograd the forward output tuple is (model_out, ..., saved_for_bw...).
    Only the first element is the actual model output; the rest are activations
    saved for backward.  We store just that one for validation.
    """
    tensors: dict[str, torch.Tensor] = {}

    if kind == "forward" and capture:
        if capture.forward_real_inputs and capture.forward_graphs:
            fg = capture.forward_graphs[0]
            phs = [n for n in fg.graph_module.graph.nodes if n.op == "placeholder"]
            for ph, t in zip(phs, capture.forward_real_inputs):
                if isinstance(t, torch.Tensor):
                    tensors[ph.name] = t

        if capture.forward_real_output is not None:
            fg = capture.forward_graphs[0]
            out_nodes = [n for n in fg.graph_module.graph.nodes if n.op == "output"]
            if out_nodes:
                out_node = out_nodes[0]
                out_args = out_node.args[0] if out_node.args else ()
                if isinstance(out_args, (tuple, list)):
                    first_arg = out_args[0] if out_args else None
                else:
                    first_arg = out_args

                real_out = capture.forward_real_output
                if isinstance(real_out, (tuple, list)):
                    first_val = real_out[0] if real_out else None
                else:
                    first_val = real_out

                if (first_arg is not None and hasattr(first_arg, 'name')
                        and isinstance(first_val, torch.Tensor)):
                    tensors[first_arg.name] = first_val

    elif kind == "backward" and capture:
        pass

    n_ph = sum(1 for n in gm.graph.nodes if n.op == "placeholder")
    n_out = len(tensors) - n_ph if len(tensors) > n_ph else 0
    logger.info(f"  inputs_only: {n_ph} placeholders + {n_out} output(s) = {len(tensors)} tensors")
    return tensors


def _collect_tensors(
    gm: GraphModule,
    groups: list[OpGroup],
    capture: AtenCapture | None = None,
    kind: str = "forward",
) -> dict[str, torch.Tensor]:
    """Collect all tensors referenced by the groups.

    Uses pre-recorded tensors from *capture* if available, falling back to a
    fresh interpreter run for any missing names. The *kind* parameter selects
    forward vs backward intermediates to prevent name collisions.
    """
    # Gather all tensor names we need
    needed: set[str] = set()
    for g in groups:
        needed.update(g.external_inputs)
        needed.update(g.external_outputs)
        for op in g.ops:
            needed.update(op.input_names)
            needed.update(op.output_names)

    # Use only the intermediates for the requested graph kind.
    recorded: dict[str, torch.Tensor] = {}
    if kind == "forward":
        if capture and capture.forward_intermediates:
            recorded.update(capture.forward_intermediates)
        if capture and capture.forward_real_inputs and capture.forward_graphs:
            fg = capture.forward_graphs[0]
            phs = [n for n in fg.graph_module.graph.nodes if n.op == "placeholder"]
            for ph, t in zip(phs, capture.forward_real_inputs):
                if isinstance(t, torch.Tensor):
                    recorded.setdefault(ph.name, t)
    elif kind == "backward":
        if capture and capture.backward_intermediates:
            recorded.update(capture.backward_intermediates)
        if capture and capture.backward_real_inputs and capture.backward_graphs:
            bg = capture.backward_graphs[0]
            phs = [n for n in bg.graph_module.graph.nodes if n.op == "placeholder"]
            for ph, t in zip(phs, capture.backward_real_inputs):
                if isinstance(t, torch.Tensor):
                    recorded.setdefault(ph.name, t)

    if needed.issubset(recorded.keys()):
        return {k: recorded[k] for k in needed if k in recorded}

    # Re-run the interpreter to fill in missing values (e.g. view ops that
    # were skipped during recording).  Use real inputs when available to
    # avoid producing garbage from random/materialized inputs.
    # IMPORTANT: only *add* missing keys — never overwrite already-recorded
    # values which came from the real execution.
    graphs = (capture.forward_graphs if kind == "forward"
              else capture.backward_graphs) if capture else []
    real_inputs = (capture.forward_real_inputs if kind == "forward"
                   else capture.backward_real_inputs) if capture else None
    if graphs:
        ag = graphs[0]
        try:
            if real_inputs:
                inputs = list(real_inputs)
            else:
                from torch_graph.tensor_dump import _materialize_inputs
                inputs = _materialize_inputs(ag.example_inputs)
            interp = _RecordInterp(ag.graph_module, skip_views=False)
            interp.run(*inputs)
            for k, v in interp.recorded.items():
                recorded.setdefault(k, v)
        except Exception as e:
            logger.warning("Tensor collection via interpreter failed: %s: %s", type(e).__name__, e)

    return {k: recorded[k] for k in needed if k in recorded}


# -----------------------------------------------------------------------------
# Dtype conversion for HDF5 (numpy doesn't support all torch dtypes)
# -----------------------------------------------------------------------------

# Dtypes that numpy handles natively
# float16 excluded: stored as f32 with blosc compression for viewer compatibility
_NUMPY_NATIVE = {
    torch.float32, torch.float64,
    torch.int8, torch.int16, torch.int32, torch.int64,
    torch.uint8, torch.bool,
    torch.complex64, torch.complex128,
}


def _torch_to_numpy(t: torch.Tensor) -> tuple:
    """Convert a torch tensor to numpy, handling exotic dtypes.

    Returns (np_array, stored_as_note) where stored_as_note is None for
    native dtypes, or a string describing the conversion.

    bfloat16 and float16 are stored as float32 — the bottom mantissa bits
    are zero, which makes blosc byte-shuffle compression very effective.
    """
    if t.dtype in _NUMPY_NATIVE:
        return t.numpy(), None
    if t.dtype == torch.bfloat16:
        return t.float().numpy(), "float32_from_bfloat16"
    if t.dtype == torch.float16:
        return t.float().numpy(), "float32_from_float16"
    # float8 variants: store raw bytes + metadata to reconstruct
    dtype_name = str(t.dtype)
    if "float8" in dtype_name:
        return t.view(torch.uint8).numpy(), f"raw_uint8_from_{dtype_name.replace('torch.', '')}"
    # Fallback: try float conversion, else raw bytes
    try:
        return t.float().numpy(), f"float32_from_{dtype_name.replace('torch.', '')}"
    except Exception:
        return t.contiguous().view(torch.uint8).numpy(), f"raw_uint8_from_{dtype_name.replace('torch.', '')}"


# -----------------------------------------------------------------------------
# HDF5 writer
# -----------------------------------------------------------------------------

def _set_tensor_attrs(ds, name: str, tensor_meta: dict[str, dict],
                      stats_dict: dict | None = None):
    """Set rich metadata attributes on an HDF5 dataset."""
    info = tensor_meta.get(name, {})
    ds.attrs["original_name"] = name
    _SCALAR_KEYS = ("dtype", "human_name", "alias", "source_file", "source_line",
                    "source_code", "module_path", "module_type", "target",
                    "category", "device")
    _LIST_KEYS = ("shape", "strides")
    for key in _SCALAR_KEYS:
        if key in info:
            ds.attrs[key] = info[key]
    for key in _LIST_KEYS:
        if key in info:
            # SymInts from dynamic shapes can't be stored in HDF5 — concretize
            ds.attrs[key] = [int(x) for x in info[key]]
    if "is_contiguous" in info:
        ds.attrs["is_contiguous"] = int(info["is_contiguous"])
    # Tensor statistics
    if stats_dict and name in stats_dict:
        for sk, sv in stats_dict[name].items():
            if isinstance(sv, bool):
                ds.attrs[f"stat_{sk}"] = int(sv)
            elif isinstance(sv, (int, float)):
                ds.attrs[f"stat_{sk}"] = sv


def _link_tensors(parent, names: list[str], prefix: str,
                  tensor_meta: dict[str, dict], f):
    """Link tensors into parent group. Format: input::fp32::1x16x32___q

    Uses human-readable alias if available, falls back to FX short name.
    Always flat — no subdirectories.
    """
    if not names:
        return

    tag = "input" if prefix == "input" else "output"

    for name in names:
        info = tensor_meta.get(name, {})
        shape = info.get("shape")
        dtype = info.get("dtype", "")
        alias = info.get("alias")
        display = alias if alias else _short_name(name)
        safe_src = f"tensors/{_sanitize_h5_name(_short_name(name))}"
        if safe_src in f:
            parts = [tag]
            if dtype:
                parts.append(_short_dtype(dtype))
            if shape is not None:
                parts.append("x".join(str(d) for d in shape))
            key = _sanitize_h5_name("::".join(parts) + "___" + display)
            if key in parent:
                # Collision — append original FX name to disambiguate
                key = _sanitize_h5_name("::".join(parts) + "___" + display + "." + _short_name(name))
            parent[key] = f[safe_src]


def _get_h5_compression(np_data=None, stored_as: str | None = None):
    """Return Blosc2 compression filter for HDF5 datasets, or None if unavailable.

    Skips compression for:
    - Small arrays (< 4KB) where overhead can exceed gains
    - Native float dtypes (float16/32/64) which compress poorly

    Exception: floats upcast from bfloat16/float16 have zeroed bottom bits
    and compress very well with byte shuffle (typically ~50% for bf16→f32).
    Blosc2 is supported by h5web (see silx-kit/h5web#1757).
    """
    try:
        import hdf5plugin
    except ImportError:
        return None
    if np_data is not None:
        if np_data.nbytes < 4096:
            return None
        # Native floats compress poorly, but upcast floats (bf16→f32, fp16→f32)
        # have zeroed bottom bits that byte-shuffle compresses well
        if np.issubdtype(np_data.dtype, np.floating):
            if stored_as is None or not stored_as.startswith("float32_from_"):
                return None
    return hdf5plugin.Blosc2(
        cname="lz4",
        clevel=5,
        filters=hdf5plugin.Blosc2.SHUFFLE,
    )


def _write_script_store(f, collector: dict[str, list[tuple[str, str]]]):
    """Write consolidated /scripts group with gzip-compressed fixed-length datasets.

    Each script type (replay, triton) becomes one dataset.  Each element is
    one full script stored as a fixed-length byte string so that gzip can
    actually compress the text content (variable-length strings store text
    in a heap that filters cannot reach).

    ``collector`` maps ``{"replay": [(h5_path, text), ...], "triton": [...]}``.
    """
    if not any(collector.values()):
        return
    sg = f.create_group("scripts")
    for kind, entries in collector.items():
        if not entries:
            continue
        paths, texts = zip(*entries)
        encoded = [t.encode("utf-8") for t in texts]
        max_len = max(len(e) for e in encoded)
        arr = np.array(encoded, dtype=f"S{max_len}")
        ds = sg.create_dataset(kind, data=arr, compression="gzip",
                               compression_opts=4, chunks=arr.shape)
        ds.attrs["paths"] = list(paths)


def _write_h5_groups_section(
    parent_grp, groups: list[OpGroup], tensor_meta: dict[str, dict], f,
    script_collector: dict[str, list[tuple[str, str]]] | None = None,
):
    """Write a list of OpGroups into an HDF5 group, linking to /tensors/."""
    for group in groups:
        safe_gname = _sanitize_h5_name(group.name)
        g = parent_grp.create_group(safe_gname)

        # Group attributes
        g.attrs["module_path"] = group.module_path
        g.attrs["module_type"] = group.module_type
        g.attrs["source_file"] = group.source_file
        g.attrs["source_line"] = group.source_line
        g.attrs["source_code"] = group.source_code
        g.attrs["node_names"] = group.node_names
        g.attrs["all_node_names"] = group.all_node_names
        g.attrs["num_ops"] = len(group.ops)
        if group.replay_script:
            g.attrs["replay_script"] = group.replay_script
            if script_collector is not None:
                idx = len(script_collector["replay"])
                script_collector["replay"].append((g.name, group.replay_script))
                g.attrs["_script_idx"] = idx
        if group.kernel_code:
            g.attrs["kernel_code"] = group.kernel_code
            if script_collector is not None:
                idx = len(script_collector["triton"])
                script_collector["triton"].append((g.name, group.kernel_code))
                g.attrs["_triton_idx"] = idx
        if group.kernel_type:
            g.attrs["kernel_type"] = group.kernel_type

        # Group-level inputs/outputs (flat)
        _link_tensors(g, group.external_inputs, "input", tensor_meta, f)
        _link_tensors(g, group.external_outputs, "output", tensor_meta, f)

        # Per-op breakdown (zero-padded numbers sort before input/output)
        n_digits = len(str(max(len(group.ops) - 1, 0))) if group.ops else 1
        for op_idx, op in enumerate(group.ops):
            safe_opname = _sanitize_h5_name(f"{op_idx:0{n_digits}d}_{op.name}")
            op_g = g.create_group(safe_opname)
            op_g.attrs["target"] = op.target
            op_g.attrs["name"] = op.name
            op_g.attrs["source_file"] = op.source_file
            op_g.attrs["source_line"] = op.source_line
            op_g.attrs["source_code"] = op.source_code
            op_g.attrs["module_path"] = op.module_path
            op_g.attrs["module_type"] = op.module_type
            if op.replay_script:
                op_g.attrs["replay_script"] = op.replay_script
                if script_collector is not None:
                    idx = len(script_collector["replay"])
                    script_collector["replay"].append((op_g.name, op.replay_script))
                    op_g.attrs["_script_idx"] = idx

            # Per-op inputs/outputs (flat)
            _link_tensors(op_g, op.input_names, "input", tensor_meta, f)
            _link_tensors(op_g, op.output_names, "output", tensor_meta, f)


def _write_h5_multi(
    path: str,
    sections: dict[str, list[OpGroup]],
    tensors: dict[str, torch.Tensor],
    primal_map: dict[str, str],
    tensor_meta: dict[str, dict],
    stats: bool = False,
    graph_code: str = "",
):
    """Write grouped tensors to HDF5 with hard-link aliasing.

    Sections dict maps H5 group paths to OpGroup lists. All sections share
    the same /tensors/ storage — zero duplication via hard links.
    """
    import h5py
    import json

    from torch_graph.tensor_dump import compute_tensor_stats

    with h5py.File(path, "w") as f:
        # Metadata
        meta_grp = f.create_group("_meta")
        if graph_code:
            meta_grp.attrs["graph_code"] = graph_code
        meta_grp.attrs["primal_map"] = json.dumps(primal_map)
        meta_grp.attrs["sections"] = list(sections.keys())
        total = sum(len(v) for v in sections.values())
        meta_grp.attrs["num_groups"] = total

        # Canonical tensor storage (shared across all sections)
        tensors_grp = f.create_group("tensors")
        stats_dict = compute_tensor_stats(tensors) if stats else {}

        for tname, t in tensors.items():
            short = _short_name(tname)
            safe_name = _sanitize_h5_name(short)
            try:
                data = t.detach().cpu()
            except RuntimeError:
                data = _materialize_tensor(t).cpu()

            original_dtype = str(data.dtype)
            np_data, stored_as = _torch_to_numpy(data)
            comp = _get_h5_compression(np_data, stored_as=stored_as)
            kwargs = {}
            if comp:
                kwargs["compression"] = comp
                kwargs["chunks"] = np_data.shape

            ds = tensors_grp.create_dataset(safe_name, data=np_data, **kwargs)

            _set_tensor_attrs(ds, tname, tensor_meta, stats_dict)
            ds.attrs["torch_dtype"] = original_dtype
            ds.attrs["original_name"] = tname
            if stored_as:
                ds.attrs["stored_as"] = stored_as

        # Write sections — labels become H5 paths (e.g. "groups", "forward/line")
        script_collector: dict[str, list[tuple[str, str]]] = {"replay": [], "triton": []}
        for label, groups in sections.items():
            section = f.require_group(label)
            _write_h5_groups_section(section, groups, tensor_meta, f, script_collector)

        # Consolidated script storage (gzip-compressed fixed-length bytes)
        _write_script_store(f, script_collector)


# -----------------------------------------------------------------------------
# PT writer
# -----------------------------------------------------------------------------

def _serialize_groups(groups: list[OpGroup]) -> list[dict]:
    """Convert OpGroups to serializable dicts for PT format."""
    groups_data = []
    for group in groups:
        ops_data = []
        for op in group.ops:
            d = {
                "name": op.name,
                "target": op.target,
                "input_names": op.input_names,
                "output_names": op.output_names,
                "source_file": op.source_file,
                "source_line": op.source_line,
                "source_code": op.source_code,
                "module_path": op.module_path,
                "module_type": op.module_type,
            }
            if op.replay_script:
                d["replay_script"] = op.replay_script
            ops_data.append(d)
        gd = {
            "name": group.name,
            "node_names": group.node_names,
            "all_node_names": group.all_node_names,
            "input_names": group.external_inputs,
            "output_names": group.external_outputs,
            "ops": ops_data,
            "module_path": group.module_path,
            "module_type": group.module_type,
            "source_file": group.source_file,
            "source_line": group.source_line,
            "source_code": group.source_code,
        }
        if group.replay_script:
            gd["replay_script"] = group.replay_script
        if group.kernel_code:
            gd["kernel_code"] = group.kernel_code
        if group.kernel_type:
            gd["kernel_type"] = group.kernel_type
        groups_data.append(gd)
    return groups_data


def _write_pt_multi(
    path: str,
    sections: dict[str, list[OpGroup]],
    tensors: dict[str, torch.Tensor],
    primal_map: dict[str, str],
    tensor_meta: dict[str, dict],
    stats: bool = False,
    graph_code: str = "",
):
    """Write grouped tensors to .pt format (no h5py dependency).

    Sections dict maps labels to OpGroup lists. Tensors stored once.
    """
    from torch_graph.tensor_dump import compute_tensor_stats

    # Canonical tensor storage — clone to ensure picklable, use short names
    canonical: dict[str, torch.Tensor] = {}
    name_map: dict[str, str] = {}  # short -> original
    for tname, t in tensors.items():
        short = _short_name(tname)
        name_map[short] = tname
        try:
            canonical[short] = t.detach().cpu().clone()
        except RuntimeError:
            canonical[short] = _materialize_tensor(t).cpu()

    stats_dict = compute_tensor_stats(canonical) if stats else {}

    payload: dict[str, Any] = {
        "tensors": canonical,
        "name_map": name_map,
        "primal_map": primal_map,
        "tensor_meta": tensor_meta,
        "graph_code": graph_code,
        "sections": list(sections.keys()),
    }

    for label, groups in sections.items():
        payload[label] = _serialize_groups(groups)

    if stats:
        payload["tensor_stats"] = stats_dict

    torch.save(payload, path)


_SCRIPT_PREAMBLE = "import torch\nimport operator\n\n"


def _write_scripts_dir(
    scripts_dir: str, sections: dict[str, list[OpGroup]],
) -> int:
    """Write each group's replay script as a standalone ``.py`` file.

    Returns the number of scripts written.
    """
    base = Path(scripts_dir)
    if base.exists():
        import shutil
        shutil.rmtree(base)
    base.mkdir(parents=True)

    n_written = 0
    multi_section = len(sections) > 1

    for label, groups in sections.items():
        section_dir = base / label if multi_section else base
        section_dir.mkdir(parents=True, exist_ok=True)

        for group in groups:
            if not group.replay_script:
                continue
            safe = _sanitize_h5_name(group.name) + ".py"
            body = _SCRIPT_PREAMBLE + "outputs = {}\n" + group.replay_script
            (section_dir / safe).write_text(body)
            n_written += 1

    return n_written


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def dump_grouped_tensors(
    capture: AtenCapture,
    path: str,
    *,
    group_by: str | list[str] = "line",
    which: str = "forward",
    include_params: bool = True,
    stats: bool = True,
    replay_scripts: bool = True,
    scripts_dir: str | None = None,
    inputs_only: bool = False,
    # Subset selectors
    nodes: list[str] | None = None,
    pattern: str | None = None,
    module: str | None = None,
    category: str | None = None,
    lines: str | None = None,
    closure: bool = False,
    hide_views: bool = False,
) -> list[OpGroup]:
    """Dump tensors grouped by operation sets.

    Supports .h5 (HDF5) or .pt (PyTorch) output. group_by can be a list
    (e.g. ["line", "module"]) — all groupings share the same tensor storage.
    Returns list of OpGroup objects describing the dump structure.
    """
    # Normalize group_by to a list
    strategies = [group_by] if isinstance(group_by, str) else list(group_by)

    graphs_to_process = []
    if which in ("forward", "both") and capture.forward_graphs:
        graphs_to_process.append(("forward", capture.forward_graphs[0]))
    if which in ("backward", "both") and capture.backward_graphs:
        graphs_to_process.append(("backward", capture.backward_graphs[0]))

    both = which == "both" and len(graphs_to_process) > 1

    # sections[section_label] = groups_list
    # section_label is strategy (single graph) or "kind/strategy" (both)
    sections: dict[str, list[OpGroup]] = {}
    all_tensors: dict[str, torch.Tensor] = {}
    all_primal_map: dict[str, str] = {}
    all_tensor_meta: dict[str, dict] = {}
    graph_code = ""

    for kind, ag in graphs_to_process:
        gm = ag.graph_module
        graph_code = gm.code
        src_map = capture.source_map if capture else None

        # 1. Build tensor metadata for this graph
        tensor_meta = _build_tensor_meta(gm, src_map, capture)

        # 1b. Infer human-readable aliases from source code
        aliases = _build_human_aliases(gm, src_map)
        for node_name, alias in aliases.items():
            if node_name in tensor_meta:
                tensor_meta[node_name]["alias"] = alias

        # 2. Select nodes
        selected = select_nodes(
            gm, nodes=nodes, pattern=pattern, module=module,
            category=category, lines=lines, source_map=src_map,
        )

        if not selected:
            continue

        # 3. Optionally expand closure
        if closure:
            selected = expand_closure(gm, selected)

        # 4. Build kernel mapping if any strategy needs it
        k_call_map: dict[str, str] | None = None
        k_details: dict[str, Any] | None = None
        if "triton" in strategies:
            if kind == "forward" and capture.triton_capture is not None:
                tc = capture.triton_capture
                k_details = {k.name: k for k in tc.kernels}
                k_call_map = _build_triton_call_map(capture)
            elif kind == "backward" and getattr(capture, "backward_triton_capture", None) is not None:
                tc = capture.backward_triton_capture
                k_details = {k.name: k for k in tc.kernels}
                k_call_map = _build_backward_triton_call_map(capture)

        # Build op groups for each strategy
        all_groups_flat: list[OpGroup] = []
        for strategy in strategies:
            groups = build_op_groups(gm, selected, group_by=strategy, source_map=src_map,
                                     hide_views=hide_views,
                                     kernel_map=k_call_map if strategy == "triton" else None)
            # Annotate triton groups with kernel source code
            if strategy == "triton" and k_details:
                for grp in groups:
                    call_key = k_call_map.get(grp.all_node_names[0], "") if grp.all_node_names and k_call_map else ""
                    base_kernel = call_key.rsplit("#", 1)[0] if "#" in call_key else call_key
                    kdef = k_details.get(base_kernel)
                    if kdef:
                        grp.kernel_code = kdef.source_code
                        grp.kernel_type = kdef.kernel_type

            # Generate replay scripts while we still have access to gm
            if replay_scripts:
                name_to_node = {n.name: n for n in gm.graph.nodes}
                for grp in groups:
                    grp.replay_script = _build_group_replay(
                        gm, grp.all_node_names)
                    # Per-op replay (single op as a group of one)
                    for op in grp.ops:
                        node = name_to_node.get(op.name)
                        if node:
                            produced: set[str] = set()
                            op.replay_script = _build_op_replay(node, produced)

            # Section label determines H5 folder structure
            if both or inputs_only:
                label = f"{kind}/{strategy}"
            else:
                label = strategy if len(strategies) > 1 else "groups"
            sections[label] = sections.get(label, []) + groups
            all_groups_flat.extend(groups)

        # 5. Collect tensors (once, from all groups across strategies)
        if inputs_only:
            tensors = _collect_inputs_only(gm, capture, kind=kind)
        else:
            tensors = _collect_tensors(gm, all_groups_flat, capture, kind=kind)

        # Filter out param tensors if not wanted
        if not include_params:
            param_names = set()
            phs = [n for n in gm.graph.nodes if n.op == "placeholder"]
            primal_map = _build_primal_map(gm, capture)
            for ph in phs:
                if ph.name in primal_map:
                    human = primal_map[ph.name]
                    if human.startswith("self.") and not human.startswith("input"):
                        param_names.add(ph.name)
            tensors = {k: v for k, v in tensors.items() if k not in param_names}

        all_tensors.update(tensors)
        all_primal_map.update(_build_primal_map(gm, capture))
        all_tensor_meta.update(tensor_meta)

    # Check we have something
    total_groups = sum(len(v) for v in sections.values())
    if total_groups == 0:
        logger.warning("No matching nodes found.")
        return []

    # 6. Write output
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    ext = path_obj.suffix.lower()

    if ext in (".h5", ".hdf5"):
        try:
            _write_h5_multi(str(path), sections, all_tensors,
                           all_primal_map, all_tensor_meta, stats=stats,
                           graph_code=graph_code)
        except ImportError:
            logger.warning("h5py not installed. Falling back to .pt format.")
            path = str(path_obj.with_suffix(".pt"))
            _write_pt_multi(path, sections, all_tensors,
                           all_primal_map, all_tensor_meta, stats=stats,
                           graph_code=graph_code)
    else:
        _write_pt_multi(str(path), sections, all_tensors,
                       all_primal_map, all_tensor_meta, stats=stats,
                       graph_code=graph_code)

    if scripts_dir:
        n_scripts = _write_scripts_dir(scripts_dir, sections)
        logger.info(f"Wrote {n_scripts} replay scripts -> {scripts_dir}/")

    n_tensors = len(all_tensors)
    parts = []
    for label, gs in sections.items():
        n_ops = sum(len(g.ops) for g in gs)
        parts.append(f"{label}: {len(gs)} groups ({n_ops} ops)")
    summary = ", ".join(parts)
    logger.info(f"Dumped {n_tensors} tensors [{summary}] -> {path}")

    # Return flat list
    all_flat = []
    for gs in sections.values():
        all_flat.extend(gs)
    return all_flat


def dump_model_ops(
    model: torch.nn.Module,
    *args,
    path: str = "model_ops.pt",
    group_by: str | list[str] = "line",
    which: str = "forward",
    include_params: bool = True,
    stats: bool = True,
    replay_scripts: bool = True,
    run_backward: bool = False,
    loss_fn: Callable | None = None,
    record_real_tensors: bool = True,
    # Subset selectors
    nodes: list[str] | None = None,
    pattern: str | None = None,
    module: str | None = None,
    category: str | None = None,
    lines: str | None = None,
    closure: bool = False,
    hide_views: bool = False,
    **kwargs,
) -> list[OpGroup]:
    """Convenience: capture aten graphs from a model and dump grouped tensors in one call."""
    output, capture = capture_aten_graphs(
        model, *args,
        run_backward=run_backward,
        loss_fn=loss_fn,
        record_real_tensors=record_real_tensors,
        **kwargs,
    )

    return dump_grouped_tensors(
        capture, path,
        group_by=group_by,
        which=which,
        include_params=include_params,
        stats=stats,
        replay_scripts=replay_scripts,
        nodes=nodes,
        pattern=pattern,
        module=module,
        category=category,
        lines=lines,
        closure=closure,
        hide_views=hide_views,
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def dump_cli(args=None):
    """CLI entry point for the dump subcommand."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="torch_graph dump",
        description="Dump tensors grouped by operation sets from a PyTorch model.",
    )
    parser.add_argument("script", help="Python script defining the model (or recipe)")
    parser.add_argument("-o", "--output", default="model_ops.h5",
                        help="Output path (.h5 or .pt, default: model_ops.h5)")
    parser.add_argument("--group-by", default="line",
                        help="Grouping strategy: line, module, op, triton, all. "
                             "Comma-separated for multiple (e.g., 'line,module')")
    parser.add_argument("--which", default="forward",
                        choices=["forward", "backward", "both"],
                        help="Which graph(s) to dump (default: forward)")
    parser.add_argument("--backward", action="store_true",
                        help="Capture backward pass (implies --which both)")
    parser.add_argument("--triton", action="store_true",
                        help="Capture Triton kernel mapping (required for --group-by triton)")

    # Toggle flags
    parser.add_argument("--no-params", action="store_true", help="Exclude parameter tensors")
    parser.add_argument("--no-stats", action="store_true", help="Skip tensor statistics")
    parser.add_argument("--no-replay", action="store_true", help="Skip replay script generation")
    parser.add_argument("--hide-views", action="store_true",
                        help="Omit view/reshape/getitem plumbing ops from output")
    parser.add_argument("--scripts-dir",
                        help="Also write standalone .py replay scripts to this directory")

    # Subset selectors
    parser.add_argument("--nodes", help="Comma-separated node names")
    parser.add_argument("--pattern", help="Glob pattern on node names (e.g., 'attn*')")
    parser.add_argument("--module", help="nn.Module path substring (e.g., 'blocks.0.attn')")
    parser.add_argument("--category", help="Op category (e.g., 'linear_algebra')")
    parser.add_argument("--lines", help="Source line range (e.g., 'model.py:40-50')")
    parser.add_argument("--closure", action="store_true",
                        help="Expand selection to include all upstream dependencies")
    parser.add_argument("--setup-fn", default="setup",
                        help="Name of the setup function for recipe files (default: setup)")

    parsed = parser.parse_args(args)

    # Parse and validate group-by
    gb_parts = [s.strip() for s in parsed.group_by.split(",")]
    valid = {"line", "module", "op", "all", "triton"}
    for gb in gb_parts:
        if gb not in valid:
            parser.error(f"Invalid group-by: {gb!r}. Choose from: {', '.join(sorted(valid))}")
    group_by_arg = gb_parts[0] if len(gb_parts) == 1 else gb_parts

    need_triton = parsed.triton or "triton" in gb_parts
    need_backward = parsed.backward
    which = "both" if need_backward else parsed.which

    # Detect if this is a recipe file (has setup() function) vs a plain script
    capture = None
    _is_recipe = False
    try:
        import importlib.util
        _spec = importlib.util.spec_from_file_location("_recipe_check", parsed.script)
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        _is_recipe = callable(getattr(_mod, parsed.setup_fn, None))
    except Exception as e:
        logger.debug("Recipe detection failed (%s: %s), treating as plain script", type(e).__name__, e)

    if _is_recipe:
        from torch_graph.extract import load_recipe, extract_training_step
        print(f"Loading recipe from {parsed.script} ({parsed.setup_fn})...")
        recipe = load_recipe(parsed.script, setup_fn=parsed.setup_fn)
        result = extract_training_step(
            model=recipe["model"],
            sample_args=recipe.get("sample_args", ()),
            loss_fn=recipe.get("loss_fn"),
            optimizer=recipe.get("optimizer"),
            get_batch=recipe.get("get_batch"),
            step_fn=recipe.get("step_fn"),
            steps=[0],
            record_real_tensors=True,
            triton=need_triton,
        )
        capture = result["capture"]
    else:
        from torch_graph.auto import extract_from_script
        print(f"Extracting model from {parsed.script}...")
        results = extract_from_script(
            parsed.script,
            output_dir=None,
            run_backward=need_backward,
            record_real_tensors=True,
            triton=need_triton,
            max_models=1,
        )
        if results:
            capture = results[0].capture

    if capture is None:
        print("No models found in script.")
        return
    node_list = parsed.nodes.split(",") if parsed.nodes else None

    groups = dump_grouped_tensors(
        capture, parsed.output,
        group_by=group_by_arg,
        which=which,
        include_params=not parsed.no_params,
        stats=not parsed.no_stats,
        replay_scripts=not parsed.no_replay,
        scripts_dir=parsed.scripts_dir,
        nodes=node_list,
        pattern=parsed.pattern,
        module=parsed.module,
        category=parsed.category,
        lines=parsed.lines,
        closure=parsed.closure,
        hide_views=parsed.hide_views,
    )

    if groups:
        print(f"\nGroups summary:")
        for g in groups:
            n_in = len(g.external_inputs)
            n_out = len(g.external_outputs)
            n_ops = len(g.ops)
            print(f"  {g.name}: {n_ops} ops, {n_in} inputs, {n_out} outputs")
