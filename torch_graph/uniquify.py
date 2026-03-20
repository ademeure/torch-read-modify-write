"""Unified uniquification engine operating on IR JSON.

Works on serialized IR dicts — no live PyTorch graph or H5 files needed.
Supports module-tree and source-line grouping strategies.

Usage::

    from torch_graph.uniquify import uniquify_ir

    ir = json.load(open("model_ir.json"))
    result = uniquify_ir(ir["forward"], strategy="module", depth=2, min_ops=5)
    for group in result.groups:
        print(f"{group.fn_name}: {group.template_key} — {len(group.instances)} instances")
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class GroupInstance:
    """One concrete occurrence of a UniqueGroup."""

    instance_id: str
    input_map: dict[str, str]   # generic_param → actual_node_name
    output_map: dict[str, str]  # generic_return → actual_node_name
    first_node: str             # name of first node (for call site ordering)


@dataclass
class UniqueGroup:
    """A set of structurally identical node sequences detected in the IR."""

    fn_name: str                # snake_case function name
    template_key: str           # e.g. "self.layers.*" or "model.py:74"
    group_type: str             # "module" or "source_line"
    module_type: str            # e.g. "Block" (for module groups)
    instances: list[GroupInstance]
    params: list[dict]          # [{"name": ..., "annotation": ...}, ...]
    returns: list[dict]         # [{"name": ..., "annotation": ...}, ...]
    body_nodes: list[dict]      # IR nodes with genericized names (the template)
    body_code: str              # generated Python for the function body
    all_node_names: set[str] = field(default_factory=set)


@dataclass
class UniqueIR:
    """Result of uniquification: groups + metadata for codegen."""

    groups: list[UniqueGroup]
    grouped_nodes: set[str]     # node names consumed by groups
    call_sites: dict[str, tuple[UniqueGroup, GroupInstance]]  # first_node → (group, inst)


# ---------------------------------------------------------------------------
# Structural signature (format-agnostic — works on IR dicts)
# ---------------------------------------------------------------------------


def build_structural_signature(
    ir_nodes: list[dict],
    group_names: set[str],
) -> tuple:
    """Build a hashable structural signature for a group of IR nodes.

    Node refs inside the group become relative indices ``("@", idx)``.
    Node refs outside become ``("ext", position)``.
    Literals become ``("lit", value)``.

    Two groups with identical signatures have the same op structure
    and can share a single function implementation.
    """
    external_inputs: list[str] = []
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


# ---------------------------------------------------------------------------
# Template key helpers
# ---------------------------------------------------------------------------


def _normalize_template_key(path: str) -> str:
    """Replace the numeric instance index with ``*``.

    ``"self.layers.0"`` → ``"self.layers.*"``
    ``"self.layers.0.attn"`` → ``"self.layers.*.attn"``
    """
    parts = path.split(".")
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].isdigit():
            parts[i] = "*"
            break
    return ".".join(parts)


def _extract_instance_idx(path: str) -> str | None:
    """Extract the numeric instance index from a module path.

    ``"self.layers.0.attn"`` → ``"0"``
    """
    for part in path.split("."):
        if part.isdigit():
            return part
    return None


def _instance_disambiguator(module_stack: list[dict]) -> str:
    """Build instance disambiguator from module_stack for source_line grouping.

    Returns the top-level numeric-indexed path, e.g. ``"layers.0"`` from a stack
    that starts with ``{"path": "self.layers.0", ...}``.
    """
    if not module_stack:
        return ""
    top = module_stack[0]["path"]
    # Strip "self." prefix for shorter disambiguator
    if top.startswith("self."):
        top = top[5:]
    return top


# ---------------------------------------------------------------------------
# Phase 1: Grouping (strategy-specific)
# ---------------------------------------------------------------------------


def _group_by_module(
    ir_nodes: list[dict],
) -> tuple[dict[str, dict[str, list[dict]]], dict[str, str]]:
    """Group IR nodes by module_stack template keys at all depths.

    Returns (templates, template_types) where:
    - templates[template_key][instance_idx] = [ir_node, ...]
    - template_types[template_key] = module_type string
    """
    templates: dict[str, dict[str, list[dict]]] = {}
    template_types: dict[str, str] = {}

    for node in ir_nodes:
        for entry in node.get("module_stack", []):
            path = entry["path"]
            mod_type = entry["type"]
            idx = _extract_instance_idx(path)
            if idx is None:
                continue
            template = _normalize_template_key(path)
            templates.setdefault(template, {}).setdefault(idx, []).append(node)
            if template not in template_types:
                template_types[template] = mod_type

    return templates, template_types


def _group_by_source_line(
    ir_nodes: list[dict],
) -> tuple[dict[str, dict[str, list[dict]]], dict[str, str]]:
    """Group IR nodes by source file + line + module instance.

    Returns (templates, template_types) in the same format as _group_by_module.
    """
    templates: dict[str, dict[str, list[dict]]] = {}
    template_types: dict[str, str] = {}

    for node in ir_nodes:
        src = node.get("source", {})
        mod_stack = node.get("module_stack", [])

        file = src.get("file", "")
        line = src.get("line", 0)
        mod_path = src.get("module_path", "")
        mod_type = src.get("module_type", "")

        if not file or not line:
            # Fall back to module_path if no source file/line
            if not mod_path:
                continue
            file = mod_path
            line = 0

        short_file = file.rsplit("/", 1)[-1]
        disambig = _instance_disambiguator(mod_stack)
        idx = _extract_instance_idx(disambig) or "0"

        if disambig:
            key = f"{short_file}:{line}@{disambig}"
        else:
            key = f"{short_file}:{line}"

        templates.setdefault(key, {}).setdefault(idx, []).append(node)
        if key not in template_types:
            template_types[key] = mod_type or short_file

    return templates, template_types


# ---------------------------------------------------------------------------
# Phase 2-4: Signature matching, I/O computation, code generation
# ---------------------------------------------------------------------------


def _build_usage_map(ir_nodes: list[dict]) -> dict[str, set[str]]:
    """Build node_name → set of consumer node names from IR args/kwargs."""
    usage: dict[str, set[str]] = defaultdict(set)

    def _collect_refs(v: dict, consumer: str):
        if "node" in v:
            usage[v["node"]].add(consumer)
            return
        kind = v.get("kind", "")
        if kind in ("list", "tuple"):
            for item in v.get("items", []):
                _collect_refs(item, consumer)
        elif kind == "dict":
            for item in v.get("items", []):
                _collect_refs(item["key"], consumer)
                _collect_refs(item["value"], consumer)
        elif kind == "slice":
            _collect_refs(v["start"], consumer)
            _collect_refs(v["stop"], consumer)
            _collect_refs(v["step"], consumer)

    for node in ir_nodes:
        name = node["name"]
        for arg in node.get("args", []):
            _collect_refs(arg, name)
        for _, kv in node.get("kwargs", {}).items():
            _collect_refs(kv, name)

    return dict(usage)


def _collect_external_refs(v: dict, group_names: set[str], ext_list: list[str]):
    """Collect node references that are external to a group."""
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


def _build_group(
    template: str,
    group_type: str,
    mod_type: str,
    instance_nodes: dict[str, list[dict]],  # {instance_idx: [ir_nodes]}
    instance_order: list[str],
    all_ir_nodes: list[dict],
    ir_returns: list[dict],
    name_remap: dict[str, str],
) -> UniqueGroup | None:
    """Build a UniqueGroup from structurally-matched instances.

    Computes external inputs/outputs, genericizes names, generates body code.
    """
    if not instance_order:
        return None

    tmpl_idx = instance_order[0]
    tmpl_nodes = instance_nodes[tmpl_idx]
    tmpl_names = {n["name"] for n in tmpl_nodes}
    all_node_names_set = {n["name"] for n in all_ir_nodes}

    # Usage map for finding outputs (nodes used outside the group)
    usage = _build_usage_map(all_ir_nodes)

    # ── External inputs ──
    external_inputs_ordered: list[str] = []
    for ir_node in tmpl_nodes:
        for arg in ir_node.get("args", []):
            _collect_external_refs(arg, tmpl_names, external_inputs_ordered)
        for _, kwarg_v in ir_node.get("kwargs", {}).items():
            _collect_external_refs(kwarg_v, tmpl_names, external_inputs_ordered)

    # ── Outputs: template nodes used outside the group ──
    tmpl_outputs: list[str] = []
    for node in tmpl_nodes:
        consumers = usage.get(node["name"], set())
        if any(c not in tmpl_names for c in consumers):
            if node["name"] not in tmpl_outputs:
                tmpl_outputs.append(node["name"])
    # Also check graph returns
    def _check_returns(val, names_set, outputs_list):
        if isinstance(val, dict):
            if "node" in val:
                if val["node"] in names_set and val["node"] not in outputs_list:
                    outputs_list.append(val["node"])
            elif "items" in val:
                for item in val["items"]:
                    _check_returns(item, names_set, outputs_list)
    for ret in ir_returns:
        _check_returns(ret, tmpl_names, tmpl_outputs)

    if not tmpl_outputs and not external_inputs_ordered:
        return None

    # ── Per-instance input/output name maps ──
    input_name_map: dict[str, dict[str, str]] = {}
    output_name_map: dict[str, dict[str, str]] = {}

    for idx in instance_order:
        inst_nodes = instance_nodes[idx]
        inst_names = {n["name"] for n in inst_nodes}

        inst_ext: list[str] = []
        for ir_node in inst_nodes:
            for arg in ir_node.get("args", []):
                _collect_external_refs(arg, inst_names, inst_ext)
            for _, kwarg_v in ir_node.get("kwargs", {}).items():
                _collect_external_refs(kwarg_v, inst_names, inst_ext)

        inp_map = {}
        for i, tmpl_ext in enumerate(external_inputs_ordered):
            local = name_remap.get(tmpl_ext, tmpl_ext)
            if i < len(inst_ext):
                inp_map[local] = name_remap.get(inst_ext[i], inst_ext[i])
        input_name_map[idx] = inp_map

        inst_outputs: list[str] = []
        for node in inst_nodes:
            consumers = usage.get(node["name"], set())
            if any(c not in inst_names for c in consumers):
                if node["name"] not in inst_outputs:
                    inst_outputs.append(node["name"])
        for ret in ir_returns:
            _check_returns(ret, inst_names, inst_outputs)

        out_map = {}
        for i, tmpl_out in enumerate(tmpl_outputs):
            local = name_remap.get(tmpl_out, tmpl_out)
            if i < len(inst_outputs):
                out_map[local] = name_remap.get(inst_outputs[i], inst_outputs[i])
        output_name_map[idx] = out_map

    # ── Genericize names ──
    raw_base = template.replace(".*", "")
    if raw_base.startswith("self."):
        raw_base = raw_base[5:]
    template_base = raw_base.replace(".", "_")
    instance_prefix = f"{template_base}_{tmpl_idx}_"

    def _genericize(name: str) -> str:
        if name.startswith(instance_prefix):
            stripped = name[len(instance_prefix):]
            if stripped and not stripped[0].isdigit():
                return stripped
        return name

    # Build param defs with generic names
    generic_params: dict[str, str] = {}
    param_defs: list[dict] = []
    used: set[str] = set()
    ir_by_name = {n["name"]: n for n in all_ir_nodes}

    for ext_name in external_inputs_ordered:
        local = name_remap.get(ext_name, ext_name)
        generic = _genericize(local)
        base = generic
        counter = 0
        while generic in used:
            counter += 1
            generic = f"{base}_{counter}"
        used.add(generic)
        generic_params[local] = generic

        annotation = ""
        ir_node = ir_by_name.get(ext_name)
        if ir_node and ir_node.get("meta"):
            meta = ir_node["meta"]
            if meta.get("shape") and meta.get("dtype"):
                dtype_name = meta["dtype"].split(".")[-1]
                dims = ", ".join(str(s) for s in meta["shape"])
                annotation = f"{dtype_name}[{dims}]"
        param_defs.append({"name": generic, "annotation": annotation})

    # Return defs
    generic_outputs: dict[str, str] = {}
    return_defs: list[dict] = []
    for out_name in tmpl_outputs:
        local = name_remap.get(out_name, out_name)
        generic_out = _genericize(local)
        base = generic_out
        counter = 0
        while generic_out in used:
            counter += 1
            generic_out = f"{base}_{counter}"
        used.add(generic_out)
        generic_outputs[local] = generic_out

        annotation = ""
        ir_node = ir_by_name.get(out_name)
        if ir_node and ir_node.get("meta"):
            meta = ir_node["meta"]
            if meta.get("shape") and meta.get("dtype"):
                dtype_name = meta["dtype"].split(".")[-1]
                dims = ", ".join(str(s) for s in meta["shape"])
                annotation = f"{dtype_name}[{dims}]"
        return_defs.append({"name": generic_out, "annotation": annotation})

    # Function name from module type
    fn_name = re.sub(r'(?<!^)(?=[A-Z])', '_', mod_type).lower()
    fn_name = re.sub(r'[^a-zA-Z0-9_]', '_', fn_name).strip('_')
    if not fn_name or fn_name[0].isdigit():
        fn_name = f"layer_{fn_name}"

    # ── Body code ──
    local_remap: dict[str, str] = {}
    for ext_name in external_inputs_ordered:
        orig = name_remap.get(ext_name, ext_name)
        local_remap[ext_name] = generic_params.get(orig, orig)
    for node in tmpl_nodes:
        orig = name_remap.get(node["name"], node["name"])
        local_remap[node["name"]] = _genericize(orig)

    local_remap_re = None
    if local_remap:
        sorted_names = sorted(local_remap.keys(), key=len, reverse=True)
        pattern = r'(?<![.])\b(?:' + '|'.join(re.escape(n) for n in sorted_names) + r')\b'
        local_remap_re = re.compile(pattern)

    body_lines: list[str] = []
    for node in tmpl_nodes:
        line = node.get("python", "")
        if not line:
            continue
        if local_remap_re is not None:
            line = local_remap_re.sub(lambda m: local_remap.get(m.group(0), m.group(0)), line)
        for sub in line.split("\n"):
            body_lines.append(f"    {sub}")

    if tmpl_outputs:
        ret_names = []
        for n in tmpl_outputs:
            orig = name_remap.get(n, n)
            ret_names.append(generic_outputs.get(orig, _genericize(orig)))
        body_lines.append(f"    return ({', '.join(ret_names)},)")

    body_code = "\n".join(body_lines)

    # ── Remap input/output maps to generic keys ──
    generic_input_map: dict[str, dict[str, str]] = {}
    for idx in instance_order:
        new_map = {}
        for orig_key, actual in input_name_map[idx].items():
            gk = generic_params.get(orig_key, orig_key)
            new_map[gk] = actual
        generic_input_map[idx] = new_map

    generic_output_map: dict[str, dict[str, str]] = {}
    for idx in instance_order:
        new_map = {}
        for orig_key, actual in output_name_map[idx].items():
            gk = generic_outputs.get(orig_key, orig_key)
            new_map[gk] = actual
        generic_output_map[idx] = new_map

    # Build instances
    all_names: set[str] = set()
    instances: list[GroupInstance] = []
    for idx in instance_order:
        nodes = instance_nodes[idx]
        for n in nodes:
            all_names.add(n["name"])
        instances.append(GroupInstance(
            instance_id=idx,
            input_map=generic_input_map[idx],
            output_map=generic_output_map[idx],
            first_node=nodes[0]["name"] if nodes else "",
        ))

    return UniqueGroup(
        fn_name=fn_name,
        template_key=template,
        group_type=group_type,
        module_type=mod_type,
        instances=instances,
        params=param_defs,
        returns=return_defs,
        body_nodes=tmpl_nodes,
        body_code=body_code,
        all_node_names=all_names,
    )


def _disambiguate_fn_names(groups: list[UniqueGroup]) -> None:
    """Ensure all function names are unique by adding numeric suffixes."""
    name_indices: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        name_indices.setdefault(g.fn_name, []).append(i)
    for name, indices in name_indices.items():
        if len(indices) <= 1:
            continue
        for j, idx in enumerate(indices):
            groups[idx].fn_name = f"{name}_{j}"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def uniquify_ir(
    ir_graph: dict,
    *,
    strategy: str = "module",
    depth: int = -1,
    min_ops: int = 0,
    name_remap: dict[str, str] | None = None,
) -> UniqueIR:
    """Detect and extract repeated patterns from an IR graph dict.

    Args:
        ir_graph: IR dict with ``"nodes"``, ``"placeholders"``, ``"returns"``
            keys (as produced by ``graph_to_ir()``).
        strategy: Grouping strategy — ``"module"`` (nn.Module hierarchy),
            ``"source_line"`` (source file + line), or ``"both"``
            (module first, source_line for leftovers).
        depth: Max depth levels.  ``1`` = top-level only, ``-1`` = all.
        min_ops: Minimum ops for a group to be extracted.
        name_remap: Optional mapping of FX node names to display names
            (from named_intermediates).
    """
    ir_nodes = ir_graph.get("nodes", [])
    ir_returns = ir_graph.get("returns", [])
    if name_remap is None:
        name_remap = {}

    # Phase 1: Build templates based on strategy
    if strategy == "module":
        templates, template_types = _group_by_module(ir_nodes)
    elif strategy == "source_line":
        templates, template_types = _group_by_source_line(ir_nodes)
    elif strategy == "both":
        # Module first at depth=1, then source_line for leftovers
        templates, template_types = _group_by_module(ir_nodes)
        # source_line templates will be added after module processing
        # (handled in the main loop below)
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}. Use 'module', 'source_line', or 'both'.")

    group_type = strategy if strategy != "both" else "module"

    # Sort by depth (shallowest first)
    sorted_keys = sorted(templates.keys(), key=lambda t: (t.count("."), t))

    # Depth limit
    if depth > 0 and sorted_keys:
        base_depth = sorted_keys[0].count(".")
        depth_limit = base_depth + depth - 1
    else:
        depth_limit = None

    groups: list[UniqueGroup] = []
    extracted: set[str] = set()

    def _process_templates(tmpls, t_types, g_type, d_limit):
        s_keys = sorted(tmpls.keys(), key=lambda t: (t.count("."), t))
        for key in s_keys:
            if d_limit is not None and key.count(".") > d_limit:
                continue

            raw_instances = tmpls[key]
            filtered: dict[str, list[dict]] = {}
            for idx, nodes in raw_instances.items():
                remaining = [n for n in nodes if n["name"] not in extracted]
                if remaining:
                    filtered[idx] = remaining

            if len(filtered) < 2:
                continue

            # Build signatures
            instance_sigs: dict[str, tuple] = {}
            for idx, nodes in filtered.items():
                group_names = {n["name"] for n in nodes}
                instance_sigs[idx] = build_structural_signature(nodes, group_names)

            if len(instance_sigs) < 2:
                continue

            # Bucket by signature (partial matching)
            sig_buckets: dict[tuple, list[str]] = defaultdict(list)
            for idx, sig in instance_sigs.items():
                sig_buckets[sig].append(idx)

            for _sig, matching in sig_buckets.items():
                if len(matching) < 2:
                    continue
                if min_ops > 0:
                    first = matching[0]
                    if len(filtered.get(first, [])) < min_ops:
                        continue

                # Order by position in the original node list
                node_positions = {n["name"]: i for i, n in enumerate(ir_nodes)}
                instance_order = sorted(matching, key=lambda idx: node_positions.get(
                    filtered[idx][0]["name"], 0) if filtered[idx] else 0)

                subset = {idx: filtered[idx] for idx in matching}
                group = _build_group(
                    key, g_type, t_types[key],
                    subset, instance_order,
                    ir_nodes, ir_returns, name_remap,
                )
                if group is not None:
                    groups.append(group)
                    extracted.update(group.all_node_names)

    _process_templates(templates, template_types, group_type, depth_limit)

    # For "both" strategy: run source_line on ungrouped nodes
    if strategy == "both":
        ungrouped = [n for n in ir_nodes if n["name"] not in extracted]
        if ungrouped:
            sl_templates, sl_types = _group_by_source_line(ungrouped)
            _process_templates(sl_templates, sl_types, "source_line", None)

    _disambiguate_fn_names(groups)

    # Build call sites map
    call_sites: dict[str, tuple[UniqueGroup, GroupInstance]] = {}
    for group in groups:
        for inst in group.instances:
            if inst.first_node:
                call_sites[inst.first_node] = (group, inst)

    return UniqueIR(
        groups=groups,
        grouped_nodes=extracted,
        call_sites=call_sites,
    )
