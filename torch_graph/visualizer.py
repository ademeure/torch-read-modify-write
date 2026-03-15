"""Visualize FX graphs as interactive HTML and JSON."""

from __future__ import annotations

import html as html_module
import json
import logging
import re
from pathlib import Path
from typing import Any

import torch
from torch.fx import Graph, GraphModule, Node

_STACK_TRACE_RE = re.compile(r'\s*File "([^"]+)", line (\d+), in (\w+)')

from torch_graph._utils import clean_self_path as _clean_self_path
from torch_graph.export import _build_primal_map
logger = logging.getLogger(__name__)

from torch_graph.capture import CapturedGraph

# Color scheme - base op types used for non-call_function nodes
OP_COLORS = {
    "placeholder": "#43D9AD",   # teal - inputs
    "get_attr": "#E99B45",      # amber - attributes
    "call_function": "#7AA2F7", # blue - default function calls
    "call_method": "#BB9AF7",   # lavender - method calls
    "call_module": "#F7768E",   # coral - module calls
    "output": "#565F89",        # muted slate - output
}

# Semantic colors for call_function nodes based on operation category
_OP_CATEGORY_KEYWORDS: list[tuple[list[str], str, str]] = [
    (["mm", "linear", "matmul", "bmm", "addmm", "convolution"],
     "#FF6B6B", "matmul"),                                      # strong red
    (["relu", "gelu", "sigmoid", "tanh", "silu", "hardswish", "leaky_relu"],
     "#9ECE6A", "activation"),                                  # green
    (["layer_norm", "batch_norm", "group_norm", "instance_norm"],
     "#E0AF68", "norm"),                                        # gold
    (["add.", "mul.", "div.", "sub.", "neg", "pow.", "rsub", "where"],
     "#BB9AF7", "elementwise"),                                 # purple
    (["softmax", "dropout", "scaled_dot"],
     "#F7768E", "attention"),                                   # coral
    (["embedding"],
     "#43D9AD", "embedding"),                                   # teal
    (["sum", "mean", "amax", "amin", "max.", "min."],
     "#FF9E64", "reduction"),                                   # orange
    (["view", "reshape", "permute", "transpose", "contiguous",
      "t.", "expand", "slice", "unsqueeze", "squeeze", "clone",
      "getitem", "split", "cat", "select", "narrow"],
     "#636DA6", "shape"),                                       # dim indigo
]


def _semantic_op_color(target_str: str) -> str:
    """Return a colour for a call_function node based on its aten op name."""
    low = target_str.lower()
    for keywords, color, _cat in _OP_CATEGORY_KEYWORDS:
        if any(k in low for k in keywords):
            return color
    return "#8B8B8B"  # grey fallback for uncategorised ops

class GraphVisualizer:
    """FX graph visualizer."""

    def __init__(self, source: CapturedGraph | GraphModule):
        # Accept CapturedGraph, AtenGraph, or raw GraphModule
        if hasattr(source, "graph_module"):
            self.gm = source.graph_module
        else:
            self.gm = source
        self.graph = self.gm.graph

    def _node_label(self, node: Node, verbose: bool = False) -> str:
        target = str(node.target)
        if hasattr(node.target, "__name__"):
            target = node.target.__name__
        elif hasattr(node.target, "name"):
            target = node.target.name()

        if node.op == "placeholder":
            label = f"INPUT: {node.name}"
        elif node.op == "output":
            label = "OUTPUT"
        elif node.op == "get_attr":
            label = f"attr: {target}"
        else:
            label = target

        if verbose and "val" in node.meta:
            val = node.meta["val"]
            if isinstance(val, torch.Tensor):
                label += f"\n{list(val.shape)}"

        return label

    # ── Interactive HTML ──────────────────────────────────────────────

    @staticmethod
    def _strip_self_prefix(s: str) -> str:
        """Strip the ``L['self'].`` prefix that dynamo adds to FQN strings."""
        if s.startswith("L['self']."):
            return s[10:]
        if s == "L['self']":
            return ""
        return s

    def _extract_source_group(self, node: Node) -> str:
        """Extract a human-readable source group from nn_module_stack metadata.

        Uses the deepest module path with <= 3 dot-separated segments so that
        e.g. ``blocks.0.attn.c_attn`` (Linear) rolls up into the parent
        ``blocks.0.attn`` (CausalSelfAttention) group.
        """
        nn_stack = (
            node.meta.get("nn_module_stack")
            or node.meta.get("fwd_nn_module_stack")
        )
        if nn_stack and isinstance(nn_stack, dict):
            best_fqn = ""
            best_type = ""
            for _key, val in nn_stack.items():
                if not isinstance(val, tuple) or len(val) < 2:
                    continue
                fqn = self._strip_self_prefix(str(val[0]))
                if not fqn:
                    continue
                parts = fqn.split(".")
                if len(parts) <= 3:
                    mod_type = val[1]
                    if isinstance(mod_type, type):
                        best_type = mod_type.__name__
                    elif isinstance(mod_type, str):
                        best_type = mod_type.rsplit(".", 1)[-1]
                    else:
                        best_type = type(mod_type).__name__
                    best_fqn = fqn
            if best_fqn:
                return f"{best_fqn} ({best_type})" if best_type else best_fqn

        # Fallback: use the actual source code line from stack_trace
        trace = node.meta.get("stack_trace", "")
        if trace:
            for line in reversed(trace.split("\n")):
                if ", in " in line and "torch" not in line.lower():
                    if (i := line.find(", code: ")) != -1:
                        code = line[i + 8:].strip()
                        if code and code != "forward":
                            return code
        return ""

    def _extract_source_info(self, node: Node, source_map: dict | None = None) -> dict:
        """Extract rich source metadata from a node for the info panel."""
        info = {}

        # Module path from nn_module_stack
        nn_stack = node.meta.get("nn_module_stack") or node.meta.get("fwd_nn_module_stack")
        if nn_stack and isinstance(nn_stack, dict):
            items = list(nn_stack.values())
            if items:
                last = items[-1]
                if isinstance(last, tuple) and len(last) >= 2:
                    info["modulePath"] = str(last[0])
                    mod_type = last[1]
                    if isinstance(mod_type, type):
                        info["moduleType"] = mod_type.__name__
                    elif isinstance(mod_type, str):
                        info["moduleType"] = mod_type.rsplit(".", 1)[-1]

        # Source function from source_fn_stack
        src_fn_key = ""
        src_fn = node.meta.get("source_fn_stack") or node.meta.get("fwd_source_fn_stack")
        if src_fn and isinstance(src_fn, (list, tuple)):
            last = src_fn[-1] if src_fn else None
            if last and isinstance(last, tuple) and len(last) >= 2:
                src_fn_key = str(last[0])
                info["sourceFn"] = src_fn_key
                fn_type = last[1]
                if isinstance(fn_type, type):
                    info["sourceFnType"] = fn_type.__name__
                elif isinstance(fn_type, str):
                    info["sourceFnType"] = fn_type.rsplit(".", 1)[-1]

        # Parse stack_trace for file/line/code
        trace = node.meta.get("stack_trace", "")
        if trace:
            lines = trace.strip().split("\n")
            for i in range(len(lines) - 1, -1, -1):
                line = lines[i]
                if "torch" in line.lower() and "test_repo" not in line.lower():
                    continue
                m = _STACK_TRACE_RE.match(line)
                if m:
                    info["sourceFile"] = m.group(1)
                    info["sourceLine"] = int(m.group(2))
                    info["sourceFnName"] = m.group(3)
                    if i + 1 < len(lines):
                        code = lines[i + 1].strip()
                        if code and not code.startswith("File "):
                            info["sourceCode"] = code
                    break

        # Fallback: look up source_fn key in the capture's source_map
        if "sourceFile" not in info and source_map and src_fn_key:
            trace_obj = source_map.get(src_fn_key)
            if not trace_obj and nn_stack:
                for _, v in nn_stack.items():
                    if isinstance(v, tuple) and len(v) >= 2:
                        mod_path = _clean_self_path(str(v[0]))
                        for k, st in source_map.items():
                            if st.module_path and mod_path and st.module_path in mod_path:
                                trace_obj = st
                                break
                    if trace_obj:
                        break
            if trace_obj:
                if trace_obj.file:
                    info["sourceFile"] = trace_obj.file
                if trace_obj.line:
                    info["sourceLine"] = trace_obj.line
                if trace_obj.code:
                    info["sourceCode"] = trace_obj.code

        return info

    def _build_graph_data(self, gm, source_map: dict | None = None,
                           tensor_stats: dict[str, dict] | None = None,
                           kernel_map: dict[str, str] | None = None,
                           kernel_details: dict | None = None,
                           primal_map: dict[str, str] | None = None) -> dict:
        """Build visualization data from a GraphModule."""
        nodes_data = []
        edges_data = []
        groups_data = []
        node_idx = {}

        param_nodes = []
        compute_nodes = []
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                param_nodes.append(node)
            else:
                compute_nodes.append(node)

        source_groups: dict[str, list[str]] = {}
        for node in compute_nodes:
            group = self._extract_source_group(node)
            if group:
                source_groups.setdefault(group, []).append(node.name)

        all_nodes = param_nodes + compute_nodes
        for i, node in enumerate(all_nodes):
            node_idx[node.name] = i
            target = self._node_label(node, verbose=True)

            meta = {}
            if "val" in node.meta:
                val = node.meta["val"]
                if isinstance(val, torch.Tensor):
                    meta["shape"] = str(list(val.shape))
                    meta["dtype"] = str(val.dtype)

            source_line = self._extract_source_group(node)
            source_info = self._extract_source_info(node, source_map=source_map)

            if node.op == "call_function":
                color = _semantic_op_color(target)
            else:
                color = OP_COLORS.get(node.op, "#999")

            nd = {
                "id": i,
                "name": node.name,
                "op": node.op,
                "target": target,
                "color": color,
                "meta": meta,
                "isParam": node.op == "placeholder",
                "sourceGroup": source_line,
                "sourceInfo": source_info,
            }
            if primal_map and node.name in primal_map:
                nd["displayName"] = primal_map[node.name]
            if kernel_map and node.name in kernel_map:
                nd["kernelGroup"] = kernel_map[node.name]
            if tensor_stats and node.name in tensor_stats:
                nd["stats"] = tensor_stats[node.name]
            nodes_data.append(nd)

        for node in all_nodes:
            for inp in node.all_input_nodes:
                if inp.name in node_idx:
                    edges_data.append({
                        "source": node_idx[inp.name],
                        "target": node_idx[node.name],
                        "isParamEdge": inp.op == "placeholder",
                    })

        # Assign ungrouped compute nodes to nearest preceding group (by node order).
        # This catches residual/skip-connection ops in backward graphs that lack
        # fwd_nn_module_stack metadata but clearly belong to the preceding block.
        if source_groups:
            node_to_group: dict[str, str] = {}
            for gname, nnames in source_groups.items():
                for nn in nnames:
                    node_to_group[nn] = gname
            last_group = None
            for node in compute_nodes:
                if node.name in node_to_group:
                    last_group = node_to_group[node.name]
                elif last_group is not None:
                    source_groups[last_group].append(node.name)

        for group_name, node_names in source_groups.items():
            groups_data.append({
                "name": group_name,
                "nodes": [node_idx[n] for n in node_names if n in node_idx],
            })

        line_groups_map: dict[str, list[str]] = {}
        line_groups_code: dict[str, str] = {}
        for node in compute_nodes:
            si = self._extract_source_info(node, source_map=source_map)
            sf = si.get("sourceFile")
            sl = si.get("sourceLine")
            if sf and sl:
                short_file = sf.rsplit("/", 1)[-1]
                key = f"{short_file}:{sl}"
                line_groups_map.setdefault(key, []).append(node.name)
                if key not in line_groups_code and si.get("sourceCode"):
                    line_groups_code[key] = si["sourceCode"]
        line_groups_data = []
        for key, node_names in line_groups_map.items():
            line_groups_data.append({
                "name": key,
                "code": line_groups_code.get(key, ""),
                "nodes": [node_idx[n] for n in node_names if n in node_idx],
            })
        # Sort by file name, then line number (chronological in source)
        def _line_sort_key(g):
            name = g["name"]
            parts = name.rsplit(":", 1)
            if len(parts) == 2:
                try:
                    return (parts[0], int(parts[1]))
                except ValueError:
                    pass
            return (name, 0)
        line_groups_data.sort(key=_line_sort_key)

        kernel_groups_data = []
        if kernel_map:
            kg_map: dict[str, list[str]] = {}
            for node in compute_nodes:
                kname = kernel_map.get(node.name)
                if kname:
                    kg_map.setdefault(kname, []).append(node.name)
            for kname, node_names in kg_map.items():
                kinfo: dict[str, Any] = {
                    "name": kname,
                    "nodes": [node_idx[n] for n in node_names if n in node_idx],
                }
                base_name = kname.rsplit("#", 1)[0] if "#" in kname else kname
                if kernel_details and base_name in kernel_details:
                    kd = kernel_details[base_name]
                    kinfo["type"] = kd.kernel_type if hasattr(kd, "kernel_type") else "triton"
                    kinfo["atenOps"] = kd.fused_aten_ops if hasattr(kd, "fused_aten_ops") else []
                    kinfo["sourceCode"] = kd.source_code if hasattr(kd, "source_code") else ""
                kernel_groups_data.append(kinfo)

        return {
            "nodes": nodes_data,
            "edges": edges_data,
            "groups": groups_data,
            "lineGroups": line_groups_data,
            "kernelGroups": kernel_groups_data,
            "code": gm.code,
            "numParams": len(param_nodes),
            "numCompute": len(compute_nodes),
        }

    def to_html(self, title: str = "FX Graph Viewer", tensor_stats: dict[str, dict] | None = None,
                 source_map: dict | None = None, backward_source=None,
                 bw_tensor_stats: dict[str, dict] | None = None,
                 kernel_map: dict[str, str] | None = None,
                 kernel_details: dict | None = None,
                 h5web: bool = False,
                 embed_h5: str | None = None,
                 primal_map: dict[str, str] | None = None) -> str:
        """Generate a self-contained interactive HTML visualization."""
        fw_data = self._build_graph_data(
            self.gm, source_map=source_map, tensor_stats=tensor_stats,
            kernel_map=kernel_map, kernel_details=kernel_details,
            primal_map=primal_map,
        )

        bw_data = None
        if backward_source:
            bw_gm = backward_source.graph_module if hasattr(backward_source, "graph_module") else backward_source
            bw_data = self._build_graph_data(bw_gm, source_map=source_map, tensor_stats=bw_tensor_stats)

        h5web_enabled = h5web or (embed_h5 is not None)
        embedded_h5_b64 = "null"
        h5_node_map: dict[str, str] = {}

        if embed_h5:
            embedded_h5_b64, h5_node_map = _encode_h5(embed_h5)

        return _load_html_template().format(
            title=html_module.escape(title),
            fw_data_json=json.dumps(fw_data),
            bw_data_json=json.dumps(bw_data),
            h5web_enabled="true" if h5web_enabled else "false",
            embedded_h5_b64=embedded_h5_b64,
            h5_node_map_json=json.dumps(h5_node_map),
        )

    def save_html(self, path: str, title: str = "FX Graph Viewer", tensor_stats: dict[str, dict] | None = None,
                   source_map: dict | None = None, backward_source=None,
                   bw_tensor_stats: dict[str, dict] | None = None,
                   kernel_map: dict[str, str] | None = None,
                   kernel_details: dict | None = None,
                   h5web: bool = False,
                   embed_h5: str | None = None,
                   primal_map: dict[str, str] | None = None) -> Path:
        """Save interactive HTML visualization."""
        p = Path(path)
        p.write_text(self.to_html(title=title, tensor_stats=tensor_stats, source_map=source_map,
                                   backward_source=backward_source, bw_tensor_stats=bw_tensor_stats,
                                   kernel_map=kernel_map, kernel_details=kernel_details,
                                   h5web=h5web, embed_h5=embed_h5, primal_map=primal_map))
        return p

    # ── JSON export ───────────────────────────────────────────────────

    @staticmethod
    def _json_target(node: Node) -> str:
        """Return a normalized string target for JSON export."""
        target = str(node.target)
        if hasattr(node.target, "__name__"):
            target = node.target.__name__
        elif hasattr(node.target, "name"):
            target = node.target.name()
        return target

    @staticmethod
    def _json_op(node: Node) -> str:
        """Return the user-facing op name for JSON export."""
        if node.op in {"call_function", "call_method", "call_module"}:
            return GraphVisualizer._json_target(node)
        return node.op

    @staticmethod
    def _json_meta(node: Node) -> dict[str, Any]:
        """Return compact per-node metadata for JSON export."""
        meta: dict[str, Any] = {}
        if "val" in node.meta:
            val = node.meta["val"]
            if isinstance(val, torch.Tensor):
                meta["shape"] = list(val.shape)
                meta["dtype"] = str(val.dtype)
        return meta

    @classmethod
    def _append_backward_user(
        cls,
        fw_nodes_by_name: dict[str, dict[str, Any]],
        fw_node: Node,
        bw_placeholder: Node,
    ) -> None:
        """Attach an explicit backward placeholder consumer to a forward node."""
        if fw_node.name not in fw_nodes_by_name:
            return
        entry = fw_nodes_by_name[fw_node.name]
        users = entry.setdefault("backward_users", [])
        if bw_placeholder.name not in users:
            users.append(bw_placeholder.name)

    @staticmethod
    def _tensor_shape(node: Node) -> list[int] | None:
        """Extract a tensor shape from node metadata when available."""
        val = node.meta.get("val")
        if isinstance(val, torch.Tensor):
            return list(val.shape)
        if isinstance(val, (tuple, list)):
            for item in val:
                if isinstance(item, torch.Tensor):
                    return list(item.shape)
        return None

    @classmethod
    def _assert_link_shape_compatible(cls, fw_node: Node, bw_placeholder: Node) -> None:
        """Fail fast if an explicit fw→bw link disagrees on tensor shape."""
        fw_shape = cls._tensor_shape(fw_node)
        bw_shape = cls._tensor_shape(bw_placeholder)
        if fw_shape is None or bw_shape is None:
            return
        if fw_shape != bw_shape:
            raise ValueError(
                "Forward/backward link shape mismatch: "
                f"{fw_node.name} {fw_shape} -> {bw_placeholder.name} {bw_shape}"
            )

    @classmethod
    def _assert_grad_shape_compatible(cls, bw_node: Node, fw_placeholder: Node) -> None:
        """Fail fast if a backward gradient result disagrees with its primal shape."""
        bw_shape = cls._tensor_shape(bw_node)
        fw_shape = cls._tensor_shape(fw_placeholder)
        if bw_shape is None or fw_shape is None:
            return
        if bw_shape != fw_shape:
            raise ValueError(
                "Backward grad shape mismatch: "
                f"{bw_node.name} {bw_shape} -> {fw_placeholder.name} {fw_shape}"
            )

    @classmethod
    def _assert_optimizer_param_shape_compatible(cls, opt_node: Node, fw_placeholder: Node) -> None:
        """Fail fast if an optimizer param/grad slot disagrees with the forward param shape."""
        opt_shape = cls._tensor_shape(opt_node)
        fw_shape = cls._tensor_shape(fw_placeholder)
        if opt_shape is None or fw_shape is None:
            return
        if opt_shape != fw_shape:
            raise ValueError(
                "Forward/optimizer link shape mismatch: "
                f"{fw_placeholder.name} {fw_shape} <-> {opt_node.name} {opt_shape}"
            )

    @staticmethod
    def _unwrap_backward_graph(source: Any) -> GraphModule | None:
        """Resolve a backward GraphModule from a graph or capture-like source."""
        if source is None:
            return None
        if hasattr(source, "backward_graphs"):
            graphs = getattr(source, "backward_graphs", None) or []
            if graphs:
                first = graphs[0]
                return first.graph_module if hasattr(first, "graph_module") else first
            return None
        return source.graph_module if hasattr(source, "graph_module") else source

    @staticmethod
    def _unwrap_optimizer_graph(source: Any) -> GraphModule | None:
        """Resolve an optimizer GraphModule from a source object."""
        if source is None:
            return None
        if hasattr(source, "optimizer_capture"):
            opt_cap = getattr(source, "optimizer_capture", None)
            if opt_cap is not None and getattr(opt_cap, "forward_graphs", None):
                first = opt_cap.forward_graphs[0]
                return first.graph_module if hasattr(first, "graph_module") else first
            return None
        if hasattr(source, "forward_graphs"):
            graphs = getattr(source, "forward_graphs", None) or []
            if graphs:
                first = graphs[0]
                return first.graph_module if hasattr(first, "graph_module") else first
            return None
        return source.graph_module if hasattr(source, "graph_module") else source

    @staticmethod
    def _get_optimizer_slot_info(source: Any) -> list[dict[str, Any]]:
        """Return optimizer placeholder metadata in placeholder order."""
        if source is None:
            return []
        if hasattr(source, "optimizer_capture"):
            opt_cap = getattr(source, "optimizer_capture", None)
            if opt_cap is not None:
                return (getattr(opt_cap, "optimizer_slot_info", None) or [[]])[0]
            return []
        return (getattr(source, "optimizer_slot_info", None) or [[]])[0]

    @classmethod
    def _append_unique(cls, entry: dict[str, Any], key: str, value: str) -> None:
        """Append a unique string to a JSON list field."""
        items = entry.setdefault(key, [])
        if value not in items:
            items.append(value)

    @staticmethod
    def _count_tensor_outputs(output: Any) -> int | None:
        """Count differentiable tensor outputs in the original forward result."""
        if output is None:
            return None
        if isinstance(output, torch.Tensor):
            return 1
        if isinstance(output, (tuple, list)):
            return sum(1 for x in output if isinstance(x, torch.Tensor))
        return 0

    @staticmethod
    def _output_args(gm: GraphModule) -> list[Any]:
        """Return the flattened top-level forward outputs."""
        out_node = next((n for n in gm.graph.nodes if n.op == "output"), None)
        if out_node is None or not out_node.args:
            return []
        out_args = out_node.args[0]
        if isinstance(out_args, (tuple, list)):
            return list(out_args)
        return [out_args]

    @staticmethod
    def _is_tensor_output(node: Node) -> bool:
        """Best-effort test for whether a forward output is tensor-valued."""
        val = node.meta.get("val")
        if isinstance(val, torch.Tensor):
            return True
        if isinstance(val, (tuple, list)):
            return any(isinstance(v, torch.Tensor) for v in val)
        # Metadata is often absent in unit tests; saved outputs are tensors in
        # the common case, so default to tensor-like rather than dropping links.
        return True

    @classmethod
    def _annotate_backward_users(
        cls,
        fw_nodes: list[dict[str, Any]],
        fw_gm: GraphModule,
        bw_gm: GraphModule,
        *,
        num_real_outputs: int | None = None,
    ) -> None:
        """Link forward outputs to the exact backward placeholders that consume them."""
        fw_nodes_by_name = {node["name"]: node for node in fw_nodes}
        fw_outputs = cls._output_args(fw_gm)
        bw_placeholders = [node for node in bw_gm.graph.nodes if node.op == "placeholder"]
        if not fw_outputs or not bw_placeholders:
            return

        if num_real_outputs is None:
            n_grad_inputs = sum(1 for p in bw_placeholders if "tangent" in p.name)
        else:
            n_grad_inputs = num_real_outputs
        n_saved = len(bw_placeholders) - n_grad_inputs
        n_mutations = max(0, len(fw_outputs) - n_grad_inputs - n_saved)

        real_outputs = fw_outputs[n_mutations:n_mutations + n_grad_inputs]
        saved_outputs = fw_outputs[n_mutations + n_grad_inputs:]

        bw_saved = bw_placeholders[:n_saved]
        bw_grad_inputs = bw_placeholders[n_saved:]

        saved_nodes = [arg for arg in saved_outputs if isinstance(arg, Node)]
        non_tensor_saved = [node for node in saved_nodes if not cls._is_tensor_output(node)]
        tensor_saved = [node for node in saved_nodes if cls._is_tensor_output(node)]

        split_at = len(non_tensor_saved)
        bw_saved_non_tensor = bw_saved[:split_at]
        bw_saved_tensor = bw_saved[split_at:]

        for fw_node, bw_ph in zip(non_tensor_saved, bw_saved_non_tensor):
            cls._assert_link_shape_compatible(fw_node, bw_ph)
            cls._append_backward_user(fw_nodes_by_name, fw_node, bw_ph)
        for fw_node, bw_ph in zip(tensor_saved, bw_saved_tensor):
            cls._assert_link_shape_compatible(fw_node, bw_ph)
            cls._append_backward_user(fw_nodes_by_name, fw_node, bw_ph)

        real_output_nodes = [arg for arg in real_outputs if isinstance(arg, Node)]
        for fw_node, bw_ph in zip(real_output_nodes, bw_grad_inputs):
            cls._assert_link_shape_compatible(fw_node, bw_ph)
            cls._append_backward_user(fw_nodes_by_name, fw_node, bw_ph)

    @staticmethod
    def _output_nodes(gm: GraphModule) -> list[Any]:
        """Return the flattened top-level output nodes/values for a graph."""
        out_node = next((n for n in gm.graph.nodes if n.op == "output"), None)
        if out_node is None or not out_node.args:
            return []
        out_args = out_node.args[0]
        if isinstance(out_args, (tuple, list)):
            return list(out_args)
        return [out_args]

    @classmethod
    def _annotate_backward_grad_targets(
        cls,
        fw_nodes: list[dict[str, Any]],
        bw_nodes: list[dict[str, Any]],
        bw_gm: GraphModule,
        fw_gm: GraphModule,
        capture: Any | None = None,
    ) -> None:
        """Link returned backward nodes to the exact forward input slot they differentiate."""
        fw_placeholders = [node for node in fw_gm.graph.nodes if node.op == "placeholder"]
        bw_returns = cls._output_nodes(bw_gm)
        if not fw_placeholders or not bw_returns:
            return

        fw_nodes_by_name = {node["name"]: node for node in fw_nodes}
        bw_nodes_by_name = {node["name"]: node for node in bw_nodes}
        primal_map = _build_primal_map(fw_gm, capture) if capture is not None else {}
        raw_primal_names = list(getattr(capture, "primal_names", []) or [])
        param_names = set(getattr(capture, "param_names", []) or [])
        buffer_names = set(getattr(capture, "buffer_names", []) or [])

        for idx, ret in enumerate(bw_returns):
            if idx >= len(fw_placeholders):
                break
            if not isinstance(ret, Node):
                continue
            fw_ph = fw_placeholders[idx]
            cls._assert_grad_shape_compatible(ret, fw_ph)
            entry = bw_nodes_by_name.get(ret.name)
            if entry is None:
                continue
            entry["grad_of"] = fw_ph.name
            if fw_ph.name in primal_map:
                entry["grad_of_display"] = primal_map[fw_ph.name]
            if idx < len(raw_primal_names) and raw_primal_names[idx] is not None:
                raw_name = raw_primal_names[idx]
                entry["grad_of_param"] = raw_name
                if raw_name in param_names:
                    entry["grad_of_kind"] = "parameter"
                elif raw_name in buffer_names:
                    entry["grad_of_kind"] = "buffer"
                else:
                    entry["grad_of_kind"] = "input"
                fw_entry = fw_nodes_by_name.get(fw_ph.name)
                if fw_entry is not None:
                    cls._append_unique(fw_entry, "backward_grads", ret.name)

    @classmethod
    def _annotate_forward_param_nodes(
        cls,
        fw_nodes: list[dict[str, Any]],
        fw_gm: GraphModule,
        capture: Any | None,
    ) -> None:
        """Add stable parameter identity fields to forward placeholders."""
        if capture is None:
            return
        fw_placeholders = [node for node in fw_gm.graph.nodes if node.op == "placeholder"]
        fw_nodes_by_name = {node["name"]: node for node in fw_nodes}
        primal_map = _build_primal_map(fw_gm, capture)
        raw_primal_names = list(getattr(capture, "primal_names", []) or [])
        param_names = set(getattr(capture, "param_names", []) or [])
        buffer_names = set(getattr(capture, "buffer_names", []) or [])

        for idx, fw_ph in enumerate(fw_placeholders):
            entry = fw_nodes_by_name.get(fw_ph.name)
            if entry is None:
                continue
            if fw_ph.name in primal_map:
                entry["param_display"] = primal_map[fw_ph.name]
            if idx >= len(raw_primal_names) or raw_primal_names[idx] is None:
                continue
            raw_name = raw_primal_names[idx]
            entry["param_name"] = raw_name
            if raw_name in param_names:
                entry["param_kind"] = "parameter"
            elif raw_name in buffer_names:
                entry["param_kind"] = "buffer"
            else:
                entry["param_kind"] = "input"

    @classmethod
    def _annotate_optimizer_links(
        cls,
        fw_nodes: list[dict[str, Any]],
        bw_nodes: list[dict[str, Any]] | None,
        opt_nodes: list[dict[str, Any]],
        fw_gm: GraphModule,
        opt_gm: GraphModule,
        capture: Any | None,
        optimizer_source: Any | None,
    ) -> None:
        """Annotate optimizer placeholders and add reverse links from fw/bw."""
        slot_info = cls._get_optimizer_slot_info(optimizer_source)
        opt_placeholders = [node for node in opt_gm.graph.nodes if node.op == "placeholder"]
        fw_placeholders = [node for node in fw_gm.graph.nodes if node.op == "placeholder"]
        fw_nodes_by_name = {node["name"]: node for node in fw_nodes}
        bw_nodes_by_name = {node["name"]: node for node in (bw_nodes or [])}
        opt_nodes_by_name = {node["name"]: node for node in opt_nodes}
        primal_map = _build_primal_map(fw_gm, capture) if capture is not None else {}

        raw_to_fw_placeholder: dict[str, Node] = {}
        raw_primal_names = list(getattr(capture, "primal_names", []) or [])
        for idx, fw_ph in enumerate(fw_placeholders):
            if idx < len(raw_primal_names) and raw_primal_names[idx] is not None:
                raw_to_fw_placeholder[raw_primal_names[idx]] = fw_ph

        raw_to_bw_grad: dict[str, str] = {}
        for node in bw_nodes or []:
            raw_name = node.get("grad_of_param")
            if raw_name:
                raw_to_bw_grad[raw_name] = node["name"]

        for idx, opt_ph in enumerate(opt_placeholders):
            if idx >= len(slot_info):
                continue
            info = slot_info[idx]
            entry = opt_nodes_by_name.get(opt_ph.name)
            if entry is None:
                continue
            role = info.get("role")
            if role:
                entry["optimizer_role"] = role
            if info.get("state_key") is not None:
                entry["optimizer_state_key"] = info["state_key"]
            raw_name = info.get("param_name")
            if not raw_name:
                continue

            entry["param_name"] = raw_name
            fw_ph = raw_to_fw_placeholder.get(raw_name)
            if fw_ph is not None:
                entry["forward_param"] = fw_ph.name
                if fw_ph.name in primal_map:
                    entry["param_display"] = primal_map[fw_ph.name]
                fw_entry = fw_nodes_by_name.get(fw_ph.name)
                if fw_entry is not None:
                    if role == "param":
                        cls._append_unique(fw_entry, "optimizer_users", opt_ph.name)
                    elif role == "grad":
                        cls._append_unique(fw_entry, "optimizer_grad_users", opt_ph.name)
                    elif role == "state":
                        cls._append_unique(fw_entry, "optimizer_state_users", opt_ph.name)
                if role in {"param", "grad"}:
                    cls._assert_optimizer_param_shape_compatible(opt_ph, fw_ph)

            bw_grad = raw_to_bw_grad.get(raw_name)
            if bw_grad:
                entry["backward_grad"] = bw_grad
                bw_entry = bw_nodes_by_name.get(bw_grad)
                if bw_entry is not None:
                    if role == "grad":
                        cls._append_unique(bw_entry, "optimizer_users", opt_ph.name)
                    elif role == "param":
                        cls._append_unique(bw_entry, "optimizer_param_users", opt_ph.name)
                    elif role == "state":
                        cls._append_unique(bw_entry, "optimizer_state_users", opt_ph.name)

    @classmethod
    def _graph_to_json(
        cls,
        gm: GraphModule,
        *,
        backward_source: CapturedGraph | GraphModule | None = None,
        capture: Any | None = None,
    ) -> dict[str, Any]:
        """Export a single graph as a JSON-serializable dict."""
        nodes = []
        for node in gm.graph.nodes:
            nodes.append({
                "name": node.name,
                "op": cls._json_op(node),
                "fx_op": node.op,
                "target": cls._json_target(node),
                "inputs": [inp.name for inp in node.all_input_nodes],
                "users": [u.name for u in node.users],
                "meta": cls._json_meta(node),
            })
        if backward_source:
            bw_gm = cls._unwrap_backward_graph(backward_source)
            n_real_outputs = cls._count_tensor_outputs(
                getattr(backward_source, "forward_real_output", None)
            )
            if bw_gm is not None:
                cls._annotate_backward_users(
                    nodes,
                    gm,
                    bw_gm,
                    num_real_outputs=n_real_outputs,
                )
        if capture is not None:
            cls._annotate_forward_param_nodes(nodes, gm, capture)
        return {"nodes": nodes, "code": gm.code}

    def to_json(
        self,
        backward_source: CapturedGraph | GraphModule | None = None,
        optimizer_source: Any | None = None,
    ) -> dict:
        """Export graph JSON, or a combined forward/backward/optimizer JSON when requested."""
        capture = backward_source if hasattr(backward_source, "primal_names") else None
        fw_data = self._graph_to_json(
            self.gm,
            backward_source=backward_source,
            capture=capture,
        )
        bw_data = None

        bw_gm = self._unwrap_backward_graph(backward_source)
        if bw_gm is not None:
            bw_data = self._graph_to_json(bw_gm)
            self._annotate_backward_grad_targets(
                fw_data["nodes"],
                bw_data["nodes"],
                bw_gm,
                self.gm,
                capture=capture,
            )

        opt_source = optimizer_source
        if opt_source is None and hasattr(backward_source, "optimizer_capture"):
            opt_source = backward_source
        opt_gm = self._unwrap_optimizer_graph(opt_source)
        opt_data = None
        if opt_gm is not None:
            opt_data = self._graph_to_json(opt_gm)
            self._annotate_optimizer_links(
                fw_data["nodes"],
                bw_data["nodes"] if bw_data is not None else None,
                opt_data["nodes"],
                self.gm,
                opt_gm,
                capture,
                opt_source,
            )

        if bw_data is None and opt_data is None:
            return fw_data

        result = {"forward": fw_data}
        if bw_data is not None:
            result["backward"] = bw_data
        if opt_data is not None:
            result["optimizer"] = opt_data
        return result

    def save_json(
        self,
        path: str,
        backward_source: CapturedGraph | GraphModule | None = None,
        optimizer_source: Any | None = None,
    ) -> Path:
        """Save graph as JSON."""
        p = Path(path)
        p.write_text(json.dumps(
            self.to_json(
                backward_source=backward_source,
                optimizer_source=optimizer_source,
            ),
            indent=2,
        ))
        return p


# ── H5 Embedding ─────────────────────────────────────────────────────

def _encode_h5(h5_path: str) -> tuple[str, dict[str, str]]:
    """Base64-encode an HDF5 file and build a node-name → H5-path mapping.

    Returns (quoted_b64_string, node_map) where the b64 string is JSON-ready
    (includes surrounding quotes) and node_map maps FX node names to H5 paths.

    The node_map contains:
    - ``node_name`` → ``/tensors/{name}`` for direct tensor access (node clicks)
    - ``{section}:{node_name}`` → in-group output link path (group clicks)
    """
    import base64

    raw = Path(h5_path).read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    quoted = json.dumps(b64)

    node_map: dict[str, str] = {}
    try:
        import h5py

        with h5py.File(h5_path, "r") as f:
            if "tensors" in f:
                for tname in f["tensors"]:
                    node_map[tname] = f"/tensors/{tname}"

            sections = []
            if "_meta" in f:
                sections = list(f["_meta"].attrs.get("sections", []))

            for sec in sections:
                if sec not in f:
                    continue
                for gname in f[sec]:
                    grp = f[sec][gname]
                    # Scan group-level and per-op subgroup links
                    containers = [(gname, grp)]
                    for child_name in grp:
                        try:
                            child = grp[child_name]
                            if isinstance(child, h5py.Group):
                                containers.append(
                                    (f"{gname}/{child_name}", child))
                        except Exception:
                            pass
                    for cpath, container in containers:
                        for item_name in container:
                            if not (item_name.startswith("output::") or
                                    item_name.startswith("input::")):
                                continue
                            try:
                                ds = container[item_name]
                                orig = ds.attrs.get("original_name", "")
                                if orig:
                                    key = f"{sec}:{orig}"
                                    if key not in node_map:
                                        node_map[key] = \
                                            f"/{sec}/{cpath}/{item_name}"
                            except Exception:
                                pass
    except Exception as e:
        logger.warning("Failed to parse HDF5 file %s: %s. Tensor links will be unavailable.", h5_path, e)

    return quoted, node_map


# ── HTML Template ─────────────────────────────────────────────────────

_html_template_cache: str | None = None


def _load_html_template() -> str:
    """Load the HTML viewer template from the external file (cached)."""
    global _html_template_cache
    if _html_template_cache is None:
        template_path = Path(__file__).parent / "viewer_template.html"
        _html_template_cache = template_path.read_text()
    return _html_template_cache
