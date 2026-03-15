"""Condense a lossless IR JSON into a minimal DAG view.

Pure JSON→JSON post-processor. No torch or live model needed.
Only keeps what's needed for dependency chains + cross-graph edges.

Usage:
    python -m torch_graph.condense_ir input.json              # prints to stdout
    python -m torch_graph.condense_ir input.json -o output.json
    python -m torch_graph.condense_ir input.json --fold        # fold trivial ops
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _extract_inputs(node: dict[str, Any]) -> list[str]:
    """Extract node name references from args and kwargs."""
    refs: list[str] = []

    def walk(v: Any) -> None:
        if isinstance(v, dict):
            if "node" in v:
                refs.append(v["node"])
            else:
                for val in v.values():
                    walk(val)
        elif isinstance(v, list):
            for item in v:
                walk(item)

    walk(node.get("args", []))
    walk(node.get("kwargs", {}))
    return refs


def _condense_source(src: dict[str, Any] | None) -> str | None:
    if not src:
        return None
    f = src.get("file", "")
    for prefix in ("/home/", "/Users/"):
        if f.startswith(prefix):
            parts = f[len(prefix):].split("/", 1)
            if len(parts) > 1:
                f = parts[1]
            break
    line = src.get("line_number", "?")
    return f"{f}:{line}" if f else None


# Only DAG-structural cross-graph edges:
# - backward_users:  fw node → bw placeholder (saved tensor flow into backward)
# - backward_grads:  fw placeholder → bw return node (which bw node computes my grad)
# - grad_of:         bw return node → fw placeholder (I'm the gradient of this input)
_DAG_EDGE_KEYS = (
    "backward_users",
    "backward_grads",
    "grad_of",
)

# Pure alias / reshape ops: output IS the input data (just reinterpreted).
# Folding removes the node, rewires consumers to point at the source, and
# transfers backward_users upstream (saved tensor = source tensor).
_FOLDABLE_TARGETS = frozenset({
    "view.default",
    "_unsafe_view.default",
    "t.default",
    "unsqueeze.default",
    "squeeze.default",
    "squeeze.dim",
    "expand.default",
    "permute.default",
    "reshape.default",
    "contiguous.default",
    "alias.default",
    "getitem",
    "slice.Tensor",
    "detach.default",
})


def _build_fold_map(nodes: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Build foldable_name → resolved_source_names mapping."""
    by_name = {n["name"]: n for n in nodes}
    foldable_names = {
        n["name"] for n in nodes
        if n.get("op") in _FOLDABLE_TARGETS
    }

    def resolve(name: str, seen: set[str] | None = None) -> list[str]:
        if seen is None:
            seen = set()
        if name in seen:
            return [name]
        seen.add(name)
        if name not in foldable_names:
            return [name]
        node = by_name.get(name)
        if node is None:
            return [name]
        inputs = node.get("inputs", [])
        if not inputs:
            return [name]  # foldable with no inputs = keep it
        result: list[str] = []
        for inp in inputs:
            result.extend(resolve(inp, seen))
        return result

    result = {}
    for name in foldable_names:
        sources = resolve(name)
        # If a node resolves only to itself (no real sources to fold into),
        # don't fold it — it's a data source (e.g. a backward placeholder).
        if sources != [name]:
            result[name] = sources
    return result


def _fold_section(
    nodes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """Remove foldable ops, rewiring inputs and annotations to their sources.

    Returns (folded_nodes, fold_map) where fold_map maps each removed node
    name to the resolved source name(s).
    """
    by_name: dict[str, dict[str, Any]] = {n["name"]: n for n in nodes}
    fold_map = _build_fold_map(nodes)
    foldable_names = set(fold_map.keys())

    # Transfer annotations from folded nodes to their sources.
    # All foldable ops are pure aliases (output IS the input), so
    # backward_users, grad_of, backward_grads all transfer upstream.
    for name in foldable_names:
        node = by_name[name]
        for key in _DAG_EDGE_KEYS:
            val = node.get(key)
            if not val:
                continue
            for src_name in fold_map[name]:
                src_node = by_name.get(src_name)
                if src_node is None:
                    continue
                if isinstance(val, list):
                    existing = src_node.setdefault(key, [])
                    for v in val:
                        if v not in existing:
                            existing.append(v)
                else:
                    src_node.setdefault(key, val)

    # Build output: skip folded nodes, rewrite inputs
    result: list[dict[str, Any]] = []
    for n in nodes:
        if n["name"] in foldable_names:
            continue
        entry = dict(n)  # shallow copy
        if "inputs" in entry:
            new_inputs: list[str] = []
            for inp in entry["inputs"]:
                if inp in fold_map:
                    for r in fold_map[inp]:
                        if r not in new_inputs:
                            new_inputs.append(r)
                else:
                    if inp not in new_inputs:
                        new_inputs.append(inp)
            entry["inputs"] = new_inputs if new_inputs else entry["inputs"]
        result.append(entry)
    return result, fold_map


def _remap_cross_graph_refs(
    nodes: list[dict[str, Any]],
    other_fold_map: dict[str, list[str]],
) -> None:
    """Remap cross-graph edge references that point to folded nodes in the other section.

    backward_users/backward_grads values are names in the OTHER graph's section.
    If those names were folded, remap to the resolved source(s).
    """
    if not other_fold_map:
        return
    for n in nodes:
        for key in ("backward_users", "backward_grads"):
            refs = n.get(key)
            if not refs or not isinstance(refs, list):
                continue
            new_refs: list[str] = []
            for ref in refs:
                if ref in other_fold_map:
                    for r in other_fold_map[ref]:
                        if r not in new_refs:
                            new_refs.append(r)
                else:
                    if ref not in new_refs:
                        new_refs.append(ref)
            n[key] = new_refs
        # grad_of is a single string, not a list
        go = n.get("grad_of")
        if go and isinstance(go, str) and go in other_fold_map:
            sources = other_fold_map[go]
            if sources:
                n["grad_of"] = sources[0]


def condense_section(
    section: dict[str, Any],
    *,
    fold: bool = False,
    include_source: bool = True,
    include_module_path: bool = True,
    include_shape: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """Condense one IR section to minimal DAG nodes.

    Returns (nodes, fold_map). fold_map is empty when fold=False.
    """
    nodes: list[dict[str, Any]] = []
    for n in section.get("placeholders", []) + section.get("nodes", []):
        entry: dict[str, Any] = {
            "name": n["name"],
            "op": "placeholder" if n.get("fx_op") == "placeholder" else n.get("target", "?"),
        }

        inputs = _extract_inputs(n)
        if inputs:
            entry["inputs"] = inputs

        for key in _DAG_EDGE_KEYS:
            if key in n:
                entry[key] = n[key]

        if include_shape:
            shape = (n.get("meta") or {}).get("shape")
            if shape:
                entry["shape"] = shape

        # source and module_path are ALWAYS last
        if include_source:
            src = _condense_source(n.get("source"))
            if src:
                entry["source"] = src
        if include_module_path:
            mod_path = (n.get("source") or {}).get("module_path")
            if mod_path:
                entry["module_path"] = mod_path

        nodes.append(entry)

    if fold:
        nodes, fold_map = _fold_section(nodes)
        return nodes, fold_map
    return nodes, {}


def condense_ir_json(
    ir_bundle: dict[str, Any],
    *,
    fold: bool = False,
    include_source: bool = True,
    include_module_path: bool = True,
    include_shape: bool = True,
) -> dict[str, Any]:
    """Condense a full IR JSON bundle into a minimal DAG view."""
    result: dict[str, Any] = {"schema": "torch_graph.condensed_ir/v1"}
    fold_maps: dict[str, dict[str, list[str]]] = {}
    opts = dict(
        fold=fold,
        include_source=include_source,
        include_module_path=include_module_path,
        include_shape=include_shape,
    )

    for section_name in ("forward", "backward", "optimizer"):
        if section_name in ir_bundle:
            nodes, fm = condense_section(ir_bundle[section_name], **opts)
            result[section_name] = nodes
            fold_maps[section_name] = fm

    # Remap cross-graph references that point to folded nodes in the other section
    if fold:
        if "forward" in result and "backward" in fold_maps:
            _remap_cross_graph_refs(result["forward"], fold_maps["backward"])
        if "backward" in result and "forward" in fold_maps:
            _remap_cross_graph_refs(result["backward"], fold_maps["forward"])

    return result


def _format_value(val: Any) -> str:
    """Format a JSON value with arrays kept on one line."""
    if isinstance(val, list):
        return "[" + ", ".join(json.dumps(v) for v in val) + "]"
    return json.dumps(val)


def _compact_json(data: dict[str, Any]) -> str:
    """Format condensed IR: each node key on its own line, arrays inline."""
    lines = ["{"]
    sections = [(k, v) for k, v in data.items() if k != "schema"]
    lines.append(f'  "schema": {json.dumps(data["schema"])},')
    for si, (section_name, nodes) in enumerate(sections):
        lines.append(f'  "{section_name}": [')
        for i, node in enumerate(nodes):
            comma = "," if i < len(nodes) - 1 else ""
            lines.append("    {")
            items = list(node.items())
            for j, (k, v) in enumerate(items):
                item_comma = "," if j < len(items) - 1 else ""
                lines.append(f'      "{k}": {_format_value(v)}{item_comma}')
            lines.append(f"    }}{comma}")
        trailing = "," if si < len(sections) - 1 else ""
        lines.append(f"  ]{trailing}")
    lines.append("}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Condense IR JSON to minimal DAG view",
    )
    parser.add_argument("input", help="Input .json IR file")
    parser.add_argument("-o", "--output", help="Output .json file (default: stdout)")
    parser.add_argument("--indent", type=int, default=None,
                        help="JSON indent (default: one node per line)")
    parser.add_argument("--fold", action="store_true",
                        help="Fold trivial ops (view, detach, getitem, etc.) "
                             "into their producers/consumers")
    parser.add_argument("--no-source", action="store_true",
                        help="Omit source file:line annotations")
    parser.add_argument("--no-module-path", action="store_true",
                        help="Omit module_path annotations")
    parser.add_argument("--no-shape", action="store_true",
                        help="Omit shape annotations")
    args = parser.parse_args()

    with open(args.input) as f:
        ir_bundle = json.load(f)

    condensed = condense_ir_json(
        ir_bundle,
        fold=args.fold,
        include_source=not args.no_source,
        include_module_path=not args.no_module_path,
        include_shape=not args.no_shape,
    )
    if args.indent is not None:
        indent = args.indent if args.indent > 0 else None
        text = json.dumps(condensed, indent=indent)
    else:
        text = _compact_json(condensed)

    if args.output:
        Path(args.output).write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
