"""Generate kernelbox-compatible test scripts from torch-graph H5 files.

Reads the grouped tensor dumps and replay scripts from an H5 file produced by
torch-graph's ``dump_grouped_tensors`` and generates standalone Python files that
work with ``kbox watch`` / ``kbox.iterate()``.
"""

from __future__ import annotations

import fnmatch
import os
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

from torch_graph._utils import h5_load_function_source, short_name as _short_name

# ── Name mapping ─────────────────────────────────────────────────────────────

_SHORT_RE = re.compile(r"^p(\d+)$")
_SHORT_D_RE = re.compile(r"^d(\d+)$")


def _long_name(name: str) -> str:
    """H5 short name -> FX name: p6 -> primals_6, d3 -> tangents_3."""
    m = _SHORT_RE.match(name)
    if m:
        return f"primals_{m.group(1)}"
    m = _SHORT_D_RE.match(name)
    if m:
        return f"tangents_{m.group(1)}"
    return name


def _dataset_node_name(ds_name: str) -> str:
    """Extract the node name from 'input::fp32::2x16x32___add_3'."""
    parts = ds_name.split("___")
    if len(parts) >= 2:
        return parts[-1]
    return ds_name


def _fx_to_h5(fx_name: str, available: set[str]) -> str | None:
    """Resolve an FX name to its H5 tensor key, or None if not stored."""
    short = _short_name(fx_name)
    if short in available:
        return short
    if fx_name in available:
        return fx_name
    return None


# ── Data structures ──────────────────────────────────────────────────────────


@dataclass
class GroupInfo:
    """Metadata for one H5 group (line/module/op)."""

    index: int
    name: str  # e.g. "005_model.py:22 [blocks.0] | qkv = self.c_attn(x)"
    path: str  # full H5 path: "forward/line/005_model.py:22 ..."
    section: str  # "forward" or "backward"
    strategy: str  # "line" or "module"
    all_node_names: list[str]
    replay_script: str
    source_code: str
    source_file: str
    source_line: int
    module_path: str
    module_type: str
    # Derived: external input/output node names (H5 short names)
    input_nodes: list[str] = field(default_factory=list)
    output_nodes: list[str] = field(default_factory=list)


# ── H5 reading ───────────────────────────────────────────────────────────────


def _available_tensors(h5_path: str) -> set[str]:
    """Return the set of tensor names stored in /tensors/."""
    with h5py.File(h5_path, "r") as f:
        return set(f["tensors"].keys())


def _parse_replay_inputs(replay_script: str) -> list[str]:
    """Extract unique input names from a replay script (FX names)."""
    return list(dict.fromkeys(re.findall(r'inputs\["(\w+)"\]', replay_script)))


def _parse_replay_out(replay_script: str) -> list[str]:
    """Extract unique output names from a replay script, ordered by last assignment."""
    # Find all output assignments in order
    all_out = re.findall(r'outputs\["(\w+)"\]\s*=', replay_script)
    # Deduplicate preserving last occurrence
    seen: dict[str, int] = {}
    for i, name in enumerate(all_out):
        seen[name] = i
    return [name for name, _ in sorted(seen.items(), key=lambda x: x[1])]


def _replay_last_out(replay_script: str) -> list[str]:
    """Output names from the final lines — the group's most meaningful results.

    When trailing lines are getitem (tuple unpacking), returns the extracted
    elements, not the tuple-producing op (which holds a tuple, not a tensor).
    """
    lines = replay_script.strip().splitlines()
    result = []
    for line in reversed(lines):
        m = re.match(r'\s*outputs\["(\w+)"\]\s*=', line)
        if m:
            if "operator.getitem" in line:
                result.append(m.group(1))
            else:
                if not result:
                    result.append(m.group(1))
                break
        else:
            break
    result.reverse()
    return result


def _group_dataset_nodes(grp: h5py.Group) -> tuple[list[str], list[str]]:
    """Return (input_nodes, output_nodes) from group-level datasets."""
    inputs, out = [], []
    for ds_name in grp.keys():
        item = grp[ds_name]
        if not isinstance(item, h5py.Dataset):
            continue
        node = _dataset_node_name(ds_name)
        if ds_name.startswith("input::"):
            inputs.append(node)
        elif ds_name.startswith("output::"):
            out.append(node)
    return inputs, out


def _discover_sections(f: h5py.File) -> list[str]:
    """Discover section paths from H5 file metadata or structure.

    Returns section paths like ``"forward/line"``, ``"backward/line"``,
    ``"groups"``, ``"line"``, etc.
    """
    if "_meta" in f:
        meta_sections = list(f["_meta"].attrs.get("sections", []))
        if meta_sections:
            return meta_sections
    # Fallback: scan top-level groups
    skip = {"tensors", "scripts", "_meta"}
    return [k for k in f.keys() if k not in skip]


def list_groups(h5_path: str, section: str | None = None, strategy: str = "line") -> list[GroupInfo]:
    """Enumerate groups in an H5 file with metadata.

    Discovers sections from file metadata rather than assuming a fixed layout.
    Works with both multi-graph (``forward/line``, ``backward/line``) and
    single-graph (``line``, ``groups``) H5 structures.
    """
    groups: list[GroupInfo] = []

    with h5py.File(h5_path, "r") as f:
        available_sections = _discover_sections(f)

        # Filter to matching sections
        candidates = []
        for sec_path in available_sections:
            # Multi-graph: "forward/line", "backward/module" etc.
            if "/" in sec_path:
                sec_kind, sec_strat = sec_path.split("/", 1)
                if section and sec_kind != section:
                    continue
                if sec_strat != strategy:
                    continue
                candidates.append((sec_path, sec_kind))
            else:
                # Single-graph: "line", "module", "groups"
                # "groups" is the default label for a single strategy + single graph
                if sec_path != strategy and sec_path != "groups":
                    continue
                # Use the requested section as kind (default to the path itself)
                candidates.append((sec_path, section or sec_path))

        for sec_path, sec_kind in candidates:
            if sec_path not in f:
                continue
            parent = f[sec_path]
            for i, name in enumerate(sorted(parent.keys())):
                grp = parent[name]
                attrs = grp.attrs
                replay = attrs.get("replay_script", "")
                if not replay:
                    continue

                ann = list(attrs.get("all_node_names", []))
                input_nodes, output_nodes = _group_dataset_nodes(grp)

                info = GroupInfo(
                    index=i,
                    name=name,
                    path=f"{sec_path}/{name}",
                    section=sec_kind,
                    strategy=strategy,
                    all_node_names=ann,
                    replay_script=replay,
                    source_code=attrs.get("source_code", ""),
                    source_file=attrs.get("source_file", ""),
                    source_line=int(attrs.get("source_line", 0)),
                    module_path=attrs.get("module_path", ""),
                    module_type=attrs.get("module_type", ""),
                    input_nodes=input_nodes,
                    output_nodes=output_nodes,
                )
                groups.append(info)

    return groups


# ── Script generation ────────────────────────────────────────────────────────


def _needs_inf(replay_script: str) -> bool:
    """Check if the replay script uses bare `inf` (e.g. `-inf`)."""
    return bool(re.search(r'\binf\b', replay_script))


def _needs_math(replay_script: str) -> bool:
    """Check if the replay script uses `math.` functions."""
    return bool(re.search(r'\bmath\.', replay_script))


def _build_input_mapping(replay_script: str, available: set[str]) -> dict[str, str]:
    """Map replay-script input names (FX) to H5 tensor keys."""
    fx_inputs = _parse_replay_inputs(replay_script)
    mapping: dict[str, str] = {}
    for fx_name in fx_inputs:
        h5_key = _fx_to_h5(fx_name, available)
        if h5_key is not None:
            mapping[fx_name] = h5_key
    return mapping


def _build_output_list(replay_script: str, available: set[str]) -> list[tuple[str, str]]:
    """Returns [(fx_name, h5_key), ...] for outputs that have stored tensors."""
    fx_out = _parse_replay_out(replay_script)
    result = []
    seen = set()
    for fx_name in fx_out:
        h5_key = _fx_to_h5(fx_name, available)
        if h5_key is not None and h5_key not in seen:
            result.append((fx_name, h5_key))
            seen.add(h5_key)
    return result


def _build_expected_out(replay_script: str, available: set[str]) -> list[tuple[str, str]]:
    """Best outputs to validate: prefers final/last, falls back to all stored."""
    # Try last out first
    last = _replay_last_out(replay_script)
    last_stored = []
    for fx_name in last:
        h5_key = _fx_to_h5(fx_name, available)
        if h5_key is not None:
            last_stored.append((fx_name, h5_key))

    if last_stored:
        return last_stored

    # Fall back to all stored out
    return _build_output_list(replay_script, available)


def _h5_relpath(h5_path: str, out_path: str | None) -> str:
    """Compute relative path from script location to H5 file."""
    if out_path is None:
        return os.path.basename(h5_path)
    return os.path.relpath(h5_path, os.path.dirname(os.path.abspath(out_path)))


def generate_group_script(
    h5_path: str,
    group: GroupInfo,
    out_path: str | None = None,
    available: set[str] | None = None,
    validate: bool = False,
) -> str:
    """Generate a kbox-watch-compatible test file for one group.

    Args:
        validate: If True, append a ``check()`` block that compares ``run()``
            output against ``expected`` using ``torch.allclose``.
    """
    if available is None:
        available = _available_tensors(h5_path)

    h5_basename = os.path.basename(h5_path)
    h5_rel = _h5_relpath(h5_path, out_path)
    input_map = _build_input_mapping(group.replay_script, available)
    output_list = _build_expected_out(group.replay_script, available)

    # All tensor keys to load
    all_keys = sorted(set(list(input_map.values()) + [h5 for _, h5 in output_list]))

    # Detect missing inputs
    fx_inputs = _parse_replay_inputs(group.replay_script)
    missing = [n for n in fx_inputs if n not in input_map]

    # Imports
    imports = ["import os", "import torch", "import operator"]
    if _needs_inf(group.replay_script):
        imports.append("inf = float('inf')")
    if _needs_math(group.replay_script):
        imports.append("import math")

    # Docstring
    doc = f'"""{group.name}\nAuto-generated from {h5_basename} — {group.section}/{group.strategy}\n"""'

    # Tensor loading — shared inline snippet
    keys_str = ", ".join(f'"{k}"' for k in all_keys)
    load_fn = h5_load_function_source()
    load_block = (
        f'_device = "cuda" if torch.cuda.is_available() else "cpu"\n'
        f'H5_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "{h5_rel}")\n'
        f'{load_fn}'
        f'_t = _load_h5(H5_PATH, keys=[{keys_str}])'
    )

    # Inputs dict
    input_lines = []
    for fx_name in fx_inputs:
        if fx_name in input_map:
            h5_key = input_map[fx_name]
            input_lines.append(f'    "{fx_name}": _t["{h5_key}"],')
    input_block = "input = {\n" + "\n".join(input_lines) + "\n}"

    # Expected out
    output_fx_names = [fx for fx, _ in output_list]
    expected_lines = [f'    _t["{h5}"],  # {fx}' for fx, h5 in output_list]
    expected_block = "expected = [\n" + "\n".join(expected_lines) + "\n]"

    # Return statement
    if output_fx_names:
        ret_items = ", ".join(f'out["{n}"]' for n in output_fx_names)
        ret_stmt = f"    return [{ret_items}]"
    else:
        ret_stmt = "    return []"

    # Run function (replay script uses inputs["x"]/outputs["x"] -> input["x"]/out["x"])
    replay_subst = group.replay_script.replace('inputs["', 'input["').replace("inputs['", "input['")
    replay_subst = replay_subst.replace('outputs["', 'out["').replace("outputs['", "out['")
    replay_indented = textwrap.indent(replay_subst, "    ")
    run_block = f"def run(input):\n    out = {{}}\n{replay_indented}\n{ret_stmt}"

    # Missing inputs warning
    missing_block = ""
    if missing:
        missing_str = ", ".join(missing)
        missing_block = (
            f"\n# WARNING: Missing tensors (not in /tensors/): {missing_str}\n"
            f"# This group cannot run in isolation. Use --section to generate a chained test.\n"
        )

    # Validation block
    check_block = ""
    if validate and output_list:
        check_block = textwrap.dedent("""\

            # ── Validation ──
            def check(atol=1e-4, rtol=1e-4):
                result = run(input)
                for i, (got, exp) in enumerate(zip(result, expected)):
                    if not torch.allclose(got, exp, atol=atol, rtol=rtol):
                        diff = (got - exp).abs().max().item()
                        print(f"MISMATCH output {i}: max_abs_diff={diff:.6e}")
                    else:
                        print(f"OK output {i}")

            if __name__ == "__main__":
                check()
            """)

    imports_block = "\n".join(imports)

    script = f"""\
#!/usr/bin/env python3
{doc}
{imports_block}

# ── Load tensors (done once) ──
{load_block}
{missing_block}
# ── Input ──
{input_block}

# ── Expected out ──
{expected_block}

# ── Replay (edit this!) ──
{run_block}
{check_block}"""

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(script)

    return script


def _collect_ops_toposorted(h5_path: str, section: str, strategy: str) -> list[tuple[str, str]]:
    """Collect all per-op replay scripts, topologically sorted by data deps.

    When multiple ops are ready, prefers ops from the same group (source line)
    as the last emitted op, then falls back to lowest group index.  This keeps
    ops from the same line of code together for readability.

    Returns [(group_name, replay_line), ...] in correct execution order.
    """
    ops_raw: list[tuple[str, str, str, list[str], int]] = []
    group_names: list[str] = []

    # Build the section path — try "section/strategy" first, fall back to bare strategy
    section_path = f"{section}/{strategy}"

    with h5py.File(h5_path, "r") as f:
        if section_path not in f:
            # Try bare strategy or "groups" for single-graph captures
            if strategy in f:
                section_path = strategy
            elif "groups" in f:
                section_path = "groups"
            else:
                return []
        parent = f[section_path]
        for gi, gk in enumerate(sorted(parent.keys())):
            group_names.append(gk)
            grp = parent[gk]
            for sk in sorted(grp.keys()):
                item = grp[sk]
                if not isinstance(item, h5py.Group):
                    continue
                rs = item.attrs.get("replay_script", "")
                if not rs:
                    continue
                out_m = re.match(r'outputs\["(\w+)"\]', rs)
                out_name = out_m.group(1) if out_m else ""
                in_names = re.findall(r'(?:inputs|outputs)\["(\w+)"\]', rs)
                in_names = [n for n in in_names if n != out_name]
                ops_raw.append((gk, rs, out_name, in_names, gi))

    name_to_producer: dict[str, int] = {}
    for i, (_, _, out_name, _, _) in enumerate(ops_raw):
        if out_name:
            name_to_producer[out_name] = i

    in_degree = [0] * len(ops_raw)
    dependents: list[list[int]] = [[] for _ in range(len(ops_raw))]
    for i, (_, _, _, in_names, _) in enumerate(ops_raw):
        for inp in in_names:
            producer = name_to_producer.get(inp)
            if producer is not None and producer != i:
                in_degree[i] += 1
                dependents[producer].append(i)

    import heapq
    ready: list[tuple[int, int, int]] = []  # (group_idx, original_pos, op_idx)
    for i, (_, _, _, _, gi) in enumerate(ops_raw):
        if in_degree[i] == 0:
            heapq.heappush(ready, (gi, i, i))

    result: list[tuple[str, str]] = []
    while ready:
        _, _, idx = heapq.heappop(ready)
        gname, rs, _, _, _ = ops_raw[idx]
        result.append((gname, rs))
        for dep in dependents[idx]:
            in_degree[dep] -= 1
            if in_degree[dep] == 0:
                _, _, _, _, dep_gi = ops_raw[dep]
                heapq.heappush(ready, (dep_gi, dep, dep))

    if len(result) < len(ops_raw):
        for i in range(len(ops_raw)):
            if in_degree[i] > 0:
                gname, rs, _, _, _ = ops_raw[i]
                result.append((gname, rs))

    return result


def generate_section_script(
    h5_path: str,
    section: str,
    strategy: str = "line",
    out_path: str | None = None,
    validate: bool = False,
) -> str:
    """Generate a test that chains all ops in a section, topologically sorted by deps.

    Args:
        validate: If True, append a ``check()`` block that compares ``run()``
            output against ``expected`` using ``torch.allclose``.
    """
    groups = list_groups(h5_path, section=section, strategy=strategy)
    if not groups:
        raise ValueError(f"No groups found for {section}/{strategy}")

    ops = _collect_ops_toposorted(h5_path, section, strategy)
    available = _available_tensors(h5_path)
    h5_basename = os.path.basename(h5_path)
    h5_rel = _h5_relpath(h5_path, out_path)

    # Collect tensor keys and expected out from groups
    all_tensor_keys: set[str] = set()
    all_input_keys: dict[str, str] = {}

    needs_inf_val = False
    needs_math_mod = False

    # Scan all op replay scripts for imports
    for _, rs in ops:
        if _needs_math(rs):
            needs_math_mod = True
        if _needs_inf(rs):
            needs_inf_val = True

    for g in groups:
        inp_map = _build_input_mapping(g.replay_script, available)
        for fx, h5 in inp_map.items():
            all_input_keys[fx] = h5
            all_tensor_keys.add(h5)

    # Expected: last output of each group for section-level validation.
    expected_pairs: list[tuple[str, str]] = []
    for g in groups:
        for fx, h5 in _build_expected_out(g.replay_script, available):
            expected_pairs.append((fx, h5))
            all_tensor_keys.add(h5)
    # Deduplicate (keep last occurrence per fx_name)
    seen_expected: dict[str, str] = {}
    for fx, h5 in expected_pairs:
        seen_expected[fx] = h5
    expected_pairs = list(seen_expected.items())

    # Build script
    doc = f'"""{section}/{strategy} — {len(ops)} ops from {len(groups)} groups\nAuto-generated from {h5_basename}\n"""'

    imports = ["import os", "import torch", "import operator"]
    if needs_inf_val:
        imports.append("inf = float('inf')")
    if needs_math_mod:
        imports.append("import math")
    imports_block = "\n".join(imports)

    expected_fx_names = [fx for fx, _ in expected_pairs]
    expected_items = ", ".join(f'"{fx}"' for fx in expected_fx_names)

    # Load tensors using shared inline loader (maps short→long names for backward)
    load_fn = h5_load_function_source(map_short_to_long=True)
    load_block = (
        f'_device = "cuda" if torch.cuda.is_available() else "cpu"\n'
        f'H5_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "{h5_rel}")\n'
        f'{load_fn}\n'
        f'input = _load_h5(H5_PATH)\n'
        f'expected = [input[k] for k in [{expected_items}] if k in input]'
    )

    # Run function — flat ops in topological order
    # Pre-pass: produced names (LHS of chain) and max LHS width
    produced: set[str] = set()
    max_lhs = 0
    for _, rs in ops:
        m = re.match(r'outputs\["(\w+)"\]\s*=', rs.strip())
        if m:
            produced.add(m.group(1))
            max_lhs = max(max_lhs, len(f'out["{m.group(1)}"]'))
    align_width = min(max_lhs, 20)

    def _ref(name: str) -> str:
        return f'out["{name}"]' if name in produced else f'input["{name}"]'

    run_lines = [
        "def run(input):",
        "    out = {}",
    ]
    expected_fx_names = [fx for fx, _ in expected_pairs]

    prev_group = None
    for gname, rs in ops:
        if gname != prev_group:
            short = gname if len(gname) <= 500 else gname[:497] + "..."
            run_lines.append(f"    # ── {short} ──")
            prev_group = gname
        line = rs.strip()
        m = re.match(r'outputs\["(\w+)"\]\s*=\s*(.*)', line)
        if m:
            oname, expr = m.group(1), m.group(2)
            # RHS: use out for produced names, input for external
            expr = re.sub(r'(?:outputs|inputs)\["(\w+)"\]', lambda m: _ref(m.group(1)), expr)
            # Fix bare inf/−inf that would cause NameError
            expr = re.sub(r"([\s,(])-inf\b", r"\1float('-inf')", expr)
            expr = re.sub(r"([\s,(])inf\b", r"\1float('inf')", expr)
            lhs = f'out["{oname}"]'
            run_lines.append(f'    {lhs:<{align_width}} = {expr}')
        else:
            run_lines.append(f"    {line}")

    run_lines.append("")
    ret_items = ", ".join(f'out["{n}"]' for n in expected_fx_names)
    run_lines.append(f"    return [{ret_items}]")

    run_block = "\n".join(run_lines)

    # Validation block
    check_block = ""
    if validate and expected_pairs:
        check_block = textwrap.dedent("""\

            # ── Validation ──
            def check(atol=1e-4, rtol=1e-4):
                result = run(input)
                for i, (got, exp) in enumerate(zip(result, expected)):
                    if not torch.allclose(got, exp, atol=atol, rtol=rtol):
                        diff = (got - exp).abs().max().item()
                        print(f"MISMATCH output {i}: max_abs_diff={diff:.6e}")
                    else:
                        print(f"OK output {i}")

            if __name__ == "__main__":
                check()
            """)

    script = f"""\
#!/usr/bin/env python3
{doc}
{imports_block}

{load_block}

{run_block}
{check_block}"""

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(script)

    return script


def _group_filename(group: GroupInfo) -> str:
    """Generate a clean filename for a group."""
    name = group.name
    m = re.match(r"^(\d+)", name)
    idx = m.group(1) if m else f"{group.index:03d}"
    clean = re.sub(r"[^\w.]+", "_", name)
    clean = re.sub(r"_+", "_", clean).strip("_")
    if len(clean) > 100:
        clean = clean[:100].rstrip("_")
    return f"test_{clean}.py"


def generate_all(
    h5_path: str,
    out_dir: str,
    section: str | None = None,
    strategy: str = "line",
    validate: bool = False,
) -> list[str]:
    """Generate test files for all groups. Returns list of generated file paths."""
    groups = list_groups(h5_path, section=section, strategy=strategy)
    available = _available_tensors(h5_path)
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    paths = []
    for g in groups:
        filename = _group_filename(g)
        out_path = str(out_dir_path / filename)
        generate_group_script(h5_path, g, out_path=out_path, available=available,
                              validate=validate)
        paths.append(out_path)

    return paths


def kbox_cli() -> None:
    """CLI entry point for ``python -m torch_graph kbox``."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="torch_graph kbox",
        description="Generate kbox test scripts from an H5 tensor dump.",
    )
    parser.add_argument("h5_file", help="Path to H5 file produced by torch-graph dump")
    parser.add_argument(
        "--section",
        choices=["forward", "backward", "both"],
        default="forward",
        help="Which section(s) to generate scripts for (default: forward)",
    )
    parser.add_argument(
        "--strategy",
        choices=["line", "module"],
        default="line",
        help="Grouping strategy (default: line)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: derived from H5 filename)",
    )
    parser.add_argument(
        "--chain",
        action="store_true",
        help="Generate a single chained section script instead of per-group scripts",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Add validation/check code to generated scripts",
    )

    args = parser.parse_args()

    h5_path = os.path.abspath(args.h5_file)
    if not os.path.isfile(h5_path):
        print(f"Error: H5 file not found: {h5_path}")
        raise SystemExit(1)

    out_dir = args.out_dir
    if out_dir is None:
        stem = Path(h5_path).stem
        out_dir = f"{stem}_kbox"

    sections = ["forward", "backward"] if args.section == "both" else [args.section]

    total_files = []
    for sec in sections:
        if args.chain:
            out_path = os.path.join(out_dir, f"{sec}_{args.strategy}_chain.py")
            try:
                generate_section_script(
                    h5_path, section=sec, strategy=args.strategy,
                    out_path=out_path, validate=args.validate,
                )
                total_files.append(out_path)
            except ValueError as e:
                print(f"  Skipping {sec}/{args.strategy}: {e}")
        else:
            paths = generate_all(
                h5_path, out_dir=out_dir, section=sec,
                strategy=args.strategy, validate=args.validate,
            )
            total_files.extend(paths)

    print(f"Generated {len(total_files)} file(s) in {out_dir}/")
    for p in total_files:
        print(f"  {p}")
