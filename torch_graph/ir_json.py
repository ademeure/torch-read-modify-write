"""Lossless JSON IR for FX/AOT graphs."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from torch_graph.export import _build_primal_map, _extract_source_group, _lookup_source_trace
from torch_graph.internal_ir import graph_to_ir, ir_graph_to_python

logger = logging.getLogger(__name__)


def _annotate_source_metadata(
    ir_graph: dict[str, Any],
    graph_module,
    source_map: dict[str, Any] | None,
) -> None:
    """Attach original source metadata to IR nodes when available."""
    nodes_by_name = {node["name"]: node for node in ir_graph.get("nodes", [])}

    if not source_map:
        return

    for node in graph_module.graph.nodes:
        if node.op in ("placeholder", "output"):
            continue
        entry = nodes_by_name.get(node.name)
        if entry is None:
            continue
        mod_path, _, src_fn = _extract_source_group(node)
        trace = _lookup_source_trace(mod_path, src_fn, source_map)
        if trace is None:
            continue
        entry["source"] = {
            "code": trace.code,
            "file": trace.file,
            "line_number": trace.line,
            "fn_name": trace.fn_name,
            "module_path": trace.module_path,
            "module_type": trace.module_type,
        }


def _all_nodes(ir_section: dict[str, Any]) -> list[dict[str, Any]]:
    """Return placeholders + nodes as a combined list for annotation lookups."""
    return ir_section.get("placeholders", []) + ir_section.get("nodes", [])


def _annotate_backward_users_by_name(
    fw_all: list[dict[str, Any]],
    bw_phs: list[dict[str, Any]],
) -> None:
    """Link forward nodes to backward placeholders using name matching.

    aot_autograd names backward placeholders identically to the forward nodes
    that produced the saved-for-backward tensors.  This is a structural
    invariant — much more reliable than positional matching with output
    layout splitting.

    Tangent placeholders (gradient inputs) are linked to the forward node
    whose output they differentiate, identified via the forward returns list.
    """
    fw_by_name = {n["name"]: n for n in fw_all}
    for bw_ph in bw_phs:
        name = bw_ph["name"]
        if "tangent" in name:
            continue
        fw_node = fw_by_name.get(name)
        if fw_node is None:
            continue
        users = fw_node.setdefault("backward_users", [])
        if name not in users:
            users.append(name)


def _annotate_cross_graph_links(result: dict[str, Any], capture: Any) -> None:
    """Add forward↔backward↔optimizer cross-graph annotations to IR sections.

    Calls the same annotation methods used by GraphVisualizer.to_json(), adding
    backward_users, backward_grads, grad_of, param_name, optimizer_role, etc.
    to the already-built IR node dicts in-place.

    Each annotation stage is independent — a failure in one (e.g. a shape
    mismatch on a dynamic-shape model) does not prevent the others from running.
    """
    from torch_graph.visualizer import GraphVisualizer as GV

    fw_section = result.get("forward")
    bw_section = result.get("backward")
    opt_section = result.get("optimizer")

    fg = capture.forward_graphs[0] if getattr(capture, "forward_graphs", None) else None
    bg = capture.backward_graphs[0] if getattr(capture, "backward_graphs", None) else None
    if fg is None:
        return
    fw_gm = fg.graph_module

    # Combined lists for name lookups (annotations are added in-place to the
    # same dicts that live in placeholders/nodes).
    fw_all = _all_nodes(fw_section) if fw_section else []

    # Forward param annotations (param_display, param_name, param_kind)
    if fw_section:
        try:
            GV._annotate_forward_param_nodes(fw_all, fw_gm, capture)
        except Exception as e:
            logger.debug(f"Forward param annotation failed: {e}")

    bw_all: list[dict[str, Any]] = []

    # Forward → backward links
    if fw_section and bw_section and bg is not None:
        bw_gm = bg.graph_module
        bw_all = _all_nodes(bw_section)
        bw_phs = bw_section.get("placeholders", [])

        # backward_users: name-based matching (reliable for all capture paths)
        try:
            _annotate_backward_users_by_name(fw_all, bw_phs)
        except Exception as e:
            logger.debug(f"Backward users annotation failed: {e}")

        # grad_of / backward_grads
        try:
            GV._annotate_backward_grad_targets(
                fw_all, bw_all, bw_gm, fw_gm, capture=capture,
            )
        except Exception as e:
            logger.debug(f"Backward grad targets annotation failed: {e}")

    # Optimizer links (optimizer_role, forward_param, backward_grad, etc.)
    if fw_section and opt_section:
        opt_cap = getattr(capture, "optimizer_capture", None)
        if opt_cap and getattr(opt_cap, "forward_graphs", None):
            try:
                opt_gm = opt_cap.forward_graphs[0].graph_module
                opt_all = _all_nodes(opt_section)
                GV._annotate_optimizer_links(
                    fw_all, bw_all, opt_all,
                    fw_gm, opt_gm, capture, opt_cap,
                )
            except Exception as e:
                logger.debug(f"Optimizer link annotation failed: {e}")


def graph_to_ir_json(
    graph_module,
    *,
    fn_name: str = "forward",
    capture: Any | None = None,
    source_map: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize a graph as lossless JSON IR."""
    primal_map = _build_primal_map(graph_module, capture) if capture is not None else None
    result = graph_to_ir(
        graph_module,
        fn_name=fn_name,
        placeholder_display_names=primal_map,
    )
    _annotate_source_metadata(result, graph_module, source_map)
    result["schema"] = "torch_graph.ir_json/v1"
    return result


def _unique_groups_to_json(
    capture: Any,
    source_map: dict[str, Any] | None,
) -> list[dict[str, Any]] | None:
    """Run uniquification on forward graph and return JSON-serializable group summaries."""
    fg = getattr(capture, "forward_graphs", None)
    if not fg:
        return None
    try:
        from torch_graph.export import (
            export_graph_to_python,
            _build_primal_map,
            _UniqueGroup,
        )

        gm = fg[0].graph_module
        primal_map = _build_primal_map(gm, capture)
        groups_out: list[_UniqueGroup] = []
        export_graph_to_python(
            gm,
            fn_name="forward",
            primal_map=primal_map,
            source_map=source_map,
            named_intermediates=True,
            uniquify=True,
            _unique_groups=groups_out,
        )
        if not groups_out:
            return None
        return [
            {
                "fn_name": g.fn_name,
                "template_key": g.template_key,
                "module_type": g.module_type,
                "num_instances": len(g.instances),
                "instance_indices": g.instance_order,
                "num_params": len(g.params),
                "num_returns": len(g.returns),
                "params": [{"name": p["name"], "annotation": p.get("annotation", "")} for p in g.params],
                "returns": [{"name": r["name"], "annotation": r.get("annotation", "")} for r in g.returns],
            }
            for g in groups_out
        ]
    except Exception as e:
        logger.debug(f"Unique group extraction for IR JSON failed: {e}")
        return None


def capture_to_ir_json(
    capture: Any,
    *,
    annotate: bool = True,
    include_unique_groups: bool = True,
) -> dict[str, Any]:
    """Serialize a full capture as lossless IR JSON.

    When *annotate* is True (default), forward↔backward↔optimizer cross-graph
    links are added (backward_users, backward_grads, grad_of, optimizer_role,
    etc.) using the same logic as GraphVisualizer.to_json().

    When *include_unique_groups* is True (default), a ``unique_groups`` key is
    added listing the repeated module groups found by hierarchical
    uniquification (template key, instance count, params, returns).
    """
    result: dict[str, Any] = {"schema": "torch_graph.ir_json_bundle/v1"}
    source_map = getattr(capture, "source_map", None)
    if getattr(capture, "forward_graphs", None):
        fg = capture.forward_graphs[0]
        result["forward"] = graph_to_ir_json(
            fg.graph_module,
            fn_name="forward",
            capture=capture,
            source_map=source_map,
        )
    if getattr(capture, "backward_graphs", None):
        bg = capture.backward_graphs[0]
        result["backward"] = graph_to_ir_json(
            bg.graph_module,
            fn_name="backward",
            source_map=source_map,
        )
    opt_cap = getattr(capture, "optimizer_capture", None)
    if opt_cap and getattr(opt_cap, "forward_graphs", None):
        og = opt_cap.forward_graphs[0]
        result["optimizer"] = graph_to_ir_json(
            og.graph_module,
            fn_name="optimizer_step",
            source_map=getattr(opt_cap, "source_map", None),
        )
        slot_info = getattr(opt_cap, "optimizer_slot_info", None)
        if slot_info:
            result["optimizer"]["slot_info"] = slot_info[0]

    if annotate and "forward" in result:
        try:
            _annotate_cross_graph_links(result, capture)
        except Exception as e:
            logger.debug(f"Cross-graph annotation failed: {e}")

    if include_unique_groups:
        ug = _unique_groups_to_json(capture, source_map)
        if ug:
            result["unique_groups"] = ug

    return result


def save_ir_json(source: Any, path: str | Path, *, annotate: bool = True) -> Path:
    """Save a graph or capture as lossless IR JSON."""
    output_path = Path(path)
    if hasattr(source, "forward_graphs"):
        payload = capture_to_ir_json(source, annotate=annotate)
    elif hasattr(source, "graph_module"):
        payload = graph_to_ir_json(source.graph_module)
    else:
        payload = graph_to_ir_json(source)

    def _default(obj):
        # SymInt/SymFloat appear when capturing dynamic shapes
        try:
            return int(obj)
        except Exception:
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    output_path.write_text(json.dumps(payload, indent=2, default=_default))
    return output_path


__all__ = [
    "capture_to_ir_json",
    "graph_to_ir_json",
    "ir_graph_to_python",
    "save_ir_json",
]
