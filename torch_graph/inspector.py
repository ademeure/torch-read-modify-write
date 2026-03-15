"""Deep inspection of captured FX graphs.

Provides structured analysis: node tables, op breakdowns, shape tracking,
dependency analysis, and subgraph extraction.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import torch
from torch.fx import Graph, GraphModule, Node

from torch_graph.capture import CapturedGraph

_SEG_RE = re.compile(r"[.:]+")


def _target_str(node) -> str:
    if hasattr(node.target, "__name__"): return node.target.__name__
    if hasattr(node.target, "name"): return node.target.name()
    return str(node.target)


@dataclass
class NodeInfo:
    """Structured info about a single FX node."""

    name: str
    op: str
    target: str
    args_summary: str
    kwargs_summary: str
    num_users: int
    meta: dict[str, Any]


class GraphInspector:
    """Inspect and analyze captured FX graphs."""

    def __init__(self, captured: CapturedGraph | GraphModule):
        # Accept CapturedGraph, AtenGraph, or raw GraphModule
        if hasattr(captured, "graph_module"):
            self.gm = captured.graph_module
            self.captured = captured
        else:
            self.gm = captured
            self.captured = None
        self.graph = self.gm.graph
        self._cached_nodes: list[NodeInfo] | None = None

    def nodes(self) -> list[NodeInfo]:
        """Structured info for every node (cached after first call)."""
        if self._cached_nodes is not None:
            return self._cached_nodes
        result = []
        for node in self.graph.nodes:
            args_str = ", ".join(_summarize_arg(a) for a in node.args)
            kwargs_str = ", ".join(f"{k}={_summarize_arg(v)}" for k, v in node.kwargs.items())
            meta = {}
            if "val" in node.meta:
                val = node.meta["val"]
                if isinstance(val, torch.Tensor):
                    meta["shape"], meta["dtype"] = list(val.shape), str(val.dtype)
                elif isinstance(val, (tuple, list)):
                    meta["val_type"], meta["val_len"] = type(val).__name__, len(val)
            if "stack_trace" in node.meta:
                meta["stack_trace"] = node.meta["stack_trace"][:200]
            result.append(NodeInfo(
                name=node.name, op=node.op, target=_target_str(node),
                args_summary=args_str, kwargs_summary=kwargs_str,
                num_users=len(node.users), meta=meta,
            ))
        self._cached_nodes = result
        return result

    def op_counts(self) -> dict[str, int]:
        """Count occurrences of each operation type."""
        counter = Counter()
        for node in self.graph.nodes:
            if node.op in ("call_function", "call_method", "call_module"):
                counter[_target_str(node)] += 1
        return dict(counter.most_common())

    def op_categories(self) -> dict[str, list[str]]:
        """Categorize ops into families (math, activation, norm, etc.).

        Uses token-level matching (split on '.', '_', '::') to avoid false
        positives from substring matching (e.g. 'exp' matching 'expand').
        """
        categories: dict[str, list[str]] = {
            "arithmetic": [], "activation": [], "normalization": [],
            "linear_algebra": [], "reduction": [], "view/reshape": [],
            "attention": [], "other": [],
        }
        # Priority-ordered keyword map: first match wins
        kw_map = [
            ({"attention", "scaled_dot_product"}, "attention"),
            ({"relu", "gelu", "silu", "sigmoid", "tanh", "softmax", "dropout"}, "activation"),
            ({"layer_norm", "batch_norm", "group_norm", "norm"}, "normalization"),
            ({"mm", "bmm", "matmul", "linear", "addmm",
              "conv", "conv1d", "conv2d", "conv3d", "convolution"}, "linear_algebra"),
            ({"sum", "mean", "max", "min", "amax", "var", "std"}, "reduction"),
            ({"view", "reshape", "permute", "transpose", "expand",
              "contiguous", "unsqueeze", "squeeze", "slice", "select"}, "view/reshape"),
            ({"add", "sub", "mul", "div", "pow", "neg", "negate",
              "exp", "log", "sqrt", "rsqrt"}, "arithmetic"),
        ]

        for node in self.graph.nodes:
            if node.op not in ("call_function", "call_method", "call_module"):
                continue
            name = _target_str(node).lower()
            # Build token set: "aten.native_layer_norm.default" → {"aten", "native_layer_norm",
            # "native", "layer", "norm", "default"} so "exp" matches "exp" but NOT "expand".
            segments = _SEG_RE.split(name)
            tokens = {tok for seg in segments for tok in [seg] + seg.split("_")}
            cat = next((c for kw, c in kw_map if kw & tokens), "other")
            categories[cat].append(node.name)

        return {k: v for k, v in categories.items() if v}

    def shapes_table(self) -> list[dict[str, Any]]:
        """Shape info for all tensor-valued nodes."""
        rows = []
        for node in self.graph.nodes:
            if "val" not in node.meta:
                continue
            val = node.meta["val"]
            if isinstance(val, torch.Tensor):
                rows.append({
                    "name": node.name,
                    "op": node.op,
                    "shape": list(val.shape),
                    "dtype": str(val.dtype),
                    "numel": val.numel(),
                })
            elif isinstance(val, (tuple, list)):
                for i, v in enumerate(val):
                    if isinstance(v, torch.Tensor):
                        rows.append({
                            "name": f"{node.name}[{i}]",
                            "op": node.op,
                            "shape": list(v.shape),
                            "dtype": str(v.dtype),
                            "numel": v.numel(),
                        })
        return rows

    def dependency_chain(self, node_name: str) -> list[str]:
        """All transitive dependencies of a node, in topological order."""
        node = _find_node(self.graph, node_name)
        visited = set()
        order = []

        def _trace(n: Node):
            if n.name in visited:
                return
            visited.add(n.name)
            for inp in n.all_input_nodes:
                _trace(inp)
            order.append(n.name)

        _trace(node)
        return order

    def users_of(self, node_name: str) -> list[str]:
        """Immediate downstream consumers of a node."""
        node = _find_node(self.graph, node_name)
        return [u.name for u in node.users]

    def find_nodes(self, pattern: str) -> list[NodeInfo]:
        """Nodes whose name or target contains the pattern (case-insensitive)."""
        pattern_lower = pattern.lower()
        return [
            ni for ni in self.nodes()
            if pattern_lower in ni.name.lower() or pattern_lower in ni.target.lower()
        ]

    def print_table(self) -> str:
        """Formatted table string of all nodes."""
        nodes = self.nodes()
        if not nodes:
            return "(empty graph)"

        lines = []
        lines.append(f"{'Name':<30} {'Op':<16} {'Target':<40} {'Users':<6} {'Shape'}")
        lines.append("-" * 120)
        for n in nodes:
            shape = ""
            if "shape" in n.meta:
                shape = str(n.meta["shape"])
            lines.append(
                f"{n.name:<30} {n.op:<16} {n.target:<40} {n.num_users:<6} {shape}"
            )
        return "\n".join(lines)


def _find_node(graph: Graph, name: str) -> Node:
    for node in graph.nodes:
        if node.name == name:
            return node
    raise KeyError(f"Node '{name}' not found in graph")


def _summarize_arg(arg) -> str:
    if isinstance(arg, Node):
        return f"%{arg.name}"
    if isinstance(arg, (list, tuple)):
        inner = ", ".join(_summarize_arg(a) for a in arg)
        return f"[{inner}]" if isinstance(arg, list) else f"({inner})"
    return repr(arg)
