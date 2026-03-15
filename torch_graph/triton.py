"""Inductor debug enrichment and aten-to-kernel mapping.

Runs ``torch.compile(..., backend="inductor")`` with tracing enabled, parses
the generated debug artifacts, and exposes the resulting kernel metadata for
attachment onto an existing graph capture.

Historically this module was framed as a standalone Triton capture path. The
runtime contract in this repo is narrower: authoritative graph identity comes
from the live aten capture path, while the Inductor debug artifacts enrich that
capture with kernel-level lowering details.
"""

from __future__ import annotations

import glob
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn


@dataclass
class TritonKernel:
    """A single captured Triton kernel."""

    name: str
    source_code: str
    fused_aten_ops: list[str]
    fused_source_nodes: list[str]
    kernel_type: str  # "triton", "extern" (cuBLAS etc.)
    aten_to_source_map: dict[str, str] = field(default_factory=dict)

    @property
    def short_name(self) -> str:
        return self.name.split("_fused_")[-1] if "_fused_" in self.name else self.name

    def __repr__(self) -> str:
        ops = ", ".join(self.fused_aten_ops[:4])
        if len(self.fused_aten_ops) > 4:
            ops += f" (+{len(self.fused_aten_ops) - 4} more)"
        return f"TritonKernel({self.name}, type={self.kernel_type}, aten=[{ops}])"


@dataclass
class TritonCapture:
    """Parsed Inductor debug artifacts and their kernel metadata.

    The class name is kept for backward compatibility. In practice this object
    represents the kernel/debug enrichment attached to an authoritative graph
    capture rather than a replacement for that graph capture.
    """

    kernels: list[TritonKernel] = field(default_factory=list)
    call_sequence: list[dict[str, Any]] = field(default_factory=list)
    aten_graph_code: str = ""
    transformed_graph_code: str = ""
    inductor_output_code: str = ""
    ir_pre_fusion: str = ""
    ir_post_fusion: str = ""
    debug_dir: str = ""

    def summary(self) -> str:
        triton_kernels = [k for k in self.kernels if k.kernel_type == "triton"]
        extern_kernels = [k for k in self.kernels if k.kernel_type == "extern"]
        lines = [
            f"Captured {len(self.kernels)} kernel call(s): "
            f"{len(triton_kernels)} Triton + {len(extern_kernels)} extern (cuBLAS etc.)"
        ]
        for k in self.kernels:
            ops = ", ".join(k.fused_aten_ops)
            src = ", ".join(k.fused_source_nodes) if k.fused_source_nodes else ""
            lines.append(f"  [{k.kernel_type:6s}] {k.name}")
            lines.append(f"           aten ops: {ops}")
            if src:
                lines.append(f"           source:   {src}")
        return "\n".join(lines)

    def op_to_kernel_map(self) -> dict[str, str]:
        """Returns a flat mapping from aten op name to kernel name."""
        result = {}
        for k in self.kernels:
            for op in k.fused_aten_ops:
                result[op] = k.name
        return result

    def kernel_by_name(self, name: str) -> TritonKernel | None:
        for k in self.kernels:
            if k.name == name:
                return k
        return None


def capture_inductor_debug(
    model_or_fn: nn.Module | Callable,
    *args,
    run_backward: bool = False,
    loss_fn: Callable | None = None,
    keep_debug_dir: bool = False,
    **kwargs,
) -> tuple[Any, TritonCapture, TritonCapture | None]:
    """Run an Inductor compilation and parse its debug artifacts.

    This is an enrichment API: callers are expected to attach the returned
    captures back onto an already-captured aten graph or use them for
    standalone kernel inspection.

    Clears the inductor cache first so debug artifacts are always fresh.
    Returns ``(output, forward_capture, backward_capture | None)``.
    """
    import torch._inductor.config as inductor_config
    import torch._inductor.codecache as codecache

    cache_dir = codecache.cache_dir()
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)

    inductor_config.trace.enabled = True

    try:
        torch.compiler.reset()
        compiled = torch.compile(model_or_fn, backend="inductor")
        if run_backward:
            output = compiled(*args, **kwargs)
            if loss_fn:
                loss = loss_fn(output)
            elif isinstance(output, (tuple, list)):
                loss = output[0].sum()
            else:
                loss = output.sum()
            loss.backward()
        else:
            with torch.no_grad():
                output = compiled(*args, **kwargs)
    finally:
        inductor_config.trace.enabled = False

    fw_capture, bw_capture = _find_and_parse_debug_artifacts_split(cache_dir)

    if not keep_debug_dir and os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)

    return output, fw_capture, bw_capture


def capture_triton_kernels(
    model_or_fn: nn.Module | Callable,
    *args,
    run_backward: bool = False,
    loss_fn: Callable | None = None,
    keep_debug_dir: bool = False,
    **kwargs,
) -> tuple[Any, TritonCapture, TritonCapture | None]:
    """Backward-compatible wrapper for :func:`capture_inductor_debug`."""
    return capture_inductor_debug(
        model_or_fn,
        *args,
        run_backward=run_backward,
        loss_fn=loss_fn,
        keep_debug_dir=keep_debug_dir,
        **kwargs,
    )


def attach_inductor_debug(
    capture: Any,
    forward_capture: TritonCapture | None,
    backward_capture: TritonCapture | None = None,
) -> Any:
    """Attach parsed Inductor debug captures onto an existing graph capture."""
    fw = forward_capture if forward_capture and forward_capture.kernels else None
    bw = backward_capture if backward_capture and backward_capture.kernels else None

    if hasattr(capture, "attach_kernel_enrichment"):
        capture.attach_kernel_enrichment(fw, bw)
        return capture

    capture.triton_capture = fw
    capture.backward_triton_capture = bw
    return capture


def enrich_capture_with_inductor_debug(
    capture: Any,
    model_or_fn: nn.Module | Callable,
    *args,
    run_backward: bool = False,
    loss_fn: Callable | None = None,
    keep_debug_dir: bool = False,
    **kwargs,
) -> tuple[TritonCapture | None, TritonCapture | None]:
    """Capture Inductor debug artifacts and attach them to ``capture``.

    Returns the attached forward/backward captures after filtering out empty
    results. The original ``capture`` object is mutated in place.
    """
    _, fw_capture, bw_capture = capture_inductor_debug(
        model_or_fn,
        *args,
        run_backward=run_backward,
        loss_fn=loss_fn,
        keep_debug_dir=keep_debug_dir,
        **kwargs,
    )
    attach_inductor_debug(capture, fw_capture, bw_capture)

    fw = fw_capture if fw_capture and fw_capture.kernels else None
    bw = bw_capture if bw_capture and bw_capture.kernels else None
    return fw, bw


def _classify_debug_dir(debug_dir: str) -> str:
    """Classify a debug dir as 'forward' or 'backward' by reading its AOT ID."""
    output_code = os.path.join(debug_dir, "output_code.py")
    if os.path.exists(output_code):
        with open(output_code) as f:
            for line in f:
                if line.startswith("# AOT ID:"):
                    if "forward" in line:
                        return "forward"
                    if "backward" in line:
                        return "backward"
                if not line.startswith("#"):
                    break
    return "unknown"


def _find_and_parse_debug_artifacts_split(
    cache_dir: str,
) -> tuple[TritonCapture, TritonCapture | None]:
    """Find .debug directories and parse into separate forward/backward captures.

    Identifies forward vs backward by the ``# AOT ID`` comment in output_code.py.
    """
    fw = TritonCapture()
    bw: TritonCapture | None = None

    if not os.path.isdir(cache_dir):
        return fw, bw

    debug_dirs = [
        d for d in glob.glob(os.path.join(cache_dir, "**", "*.debug"), recursive=True)
        if os.path.isdir(d)
    ]

    for debug_dir in debug_dirs:
        cap = TritonCapture()
        cap.debug_dir = debug_dir
        for filename, attr in [
            ("output_code.py", "inductor_output_code"),
            ("fx_graph_readable.py", "aten_graph_code"),
            ("fx_graph_transformed.py", "transformed_graph_code"),
            ("ir_pre_fusion.txt", "ir_pre_fusion"),
            ("ir_post_fusion.txt", "ir_post_fusion"),
        ]:
            path = os.path.join(debug_dir, filename)
            if os.path.exists(path):
                setattr(cap, attr, Path(path).read_text())
        if not cap.inductor_output_code:
            continue
        _parse_output_code(cap.inductor_output_code, cap)

        kind = _classify_debug_dir(debug_dir)
        if kind == "forward":
            fw = cap
        elif kind == "backward":
            bw = cap
        elif fw.inductor_output_code == "":
            fw = cap
        else:
            bw = cap

    return fw, bw


def _find_and_parse_debug_artifacts(cache_dir: str, capture: TritonCapture) -> None:
    """Find .debug directories in the inductor cache and parse them (legacy)."""
    fw, _ = _find_and_parse_debug_artifacts_split(cache_dir)
    capture.kernels = fw.kernels
    capture.call_sequence = fw.call_sequence
    capture.debug_dir = fw.debug_dir
    capture.inductor_output_code = fw.inductor_output_code
    capture.aten_graph_code = fw.aten_graph_code
    capture.transformed_graph_code = fw.transformed_graph_code
    capture.ir_pre_fusion = fw.ir_pre_fusion
    capture.ir_post_fusion = fw.ir_post_fusion


def _parse_output_code(code: str, capture: TritonCapture) -> None:
    """Parse the inductor output_code.py to extract kernels and mappings."""
    _parse_triton_kernel_definitions(code, capture)
    _parse_call_method(code, capture)


def _parse_triton_kernel_definitions(code: str, capture: TritonCapture) -> None:
    """Extract Triton kernel definitions with their aten op annotations.

    Supports two Inductor output formats:

    Legacy (async_compile):
        # Topologically Sorted Source Nodes: [linear, x], Original ATen: [aten.addmm, aten.relu]
        triton_poi_fused_addmm_relu_0 = async_compile.triton('triton_poi_fused_addmm_relu_0', '''
        ...
        ''', device_str='cuda')

    Modern (PyTorch 2.10+, inline @triton.jit):
        # Topologically Sorted Source Nodes: [pos], Original ATen: [aten.arange]
        # ... optional heuristics decorators ...
        @triton.jit
        def triton_poi_fused_arange_0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
            ...
    """
    # --- Legacy async_compile format ---
    legacy_pattern = re.compile(
        r"# Topologically Sorted Source Nodes: \[([^\]]*)\], Original ATen: \[([^\]]*)\]\n"
        r"((?:#[^\n]*\n)*)"  # optional additional comment lines (mapping, graph fragment)
        r"(\w+)\s*=\s*async_compile\.triton\('[^']+',\s*'''(.*?)'''",
        re.DOTALL,
    )

    for match in legacy_pattern.finditer(code):
        source_nodes = [s.strip() for s in match.group(1).split(",")]
        aten_ops = [s.strip() for s in match.group(2).split(",")]
        extra_comments = match.group(3)
        kernel_name = match.group(4)
        kernel_source = match.group(5).strip()

        aten_to_source = _parse_source_mapping(extra_comments)

        kernel = TritonKernel(
            name=kernel_name,
            source_code=kernel_source,
            fused_aten_ops=aten_ops,
            fused_source_nodes=source_nodes,
            kernel_type="triton",
            aten_to_source_map=aten_to_source,
        )
        capture.kernels.append(kernel)

    if capture.kernels:
        return  # Legacy format matched, skip modern format

    # --- Modern inline @triton.jit format (PyTorch 2.10+) ---
    # Two-pass approach: first find all kernel defs with @triton.jit, then
    # walk backwards to find the associated comment block.

    # Pass 1: Find all @triton.jit kernel definitions and their source
    kernel_def_pattern = re.compile(
        r"@triton\.jit\ndef (triton_\w+)\(([^)]*)\):(.*?)(?=\n@triton\.|\ntriton_helpers|\nclass |\ndef call\(|\Z)",
        re.DOTALL,
    )
    kernel_sources: dict[str, str] = {}
    kernel_positions: dict[str, int] = {}
    for m in kernel_def_pattern.finditer(code):
        kernel_sources[m.group(1)] = m.group(0).strip()
        kernel_positions[m.group(1)] = m.start()

    if not kernel_positions:
        return

    # Pass 2: For each kernel, scan backwards from its position to find the
    # "Topologically Sorted Source Nodes" comment block
    topo_pattern = re.compile(
        r"# Topologically Sorted Source Nodes: \[([^\]]*)\], Original ATen: \[([^\]]*)\]"
    )
    source_map_pattern = re.compile(r"#\s+(\S.*?)\s+=>\s+(\S.*)")

    for kname, pos in kernel_positions.items():
        # Look at the text before this kernel def.  Large kernels can have
        # 5000+ chars of triton_meta/inductor_meta decorators between the
        # "Topologically Sorted" comment and the @triton.jit line.
        preceding = code[max(0, pos - 20000):pos]
        lines = preceding.split("\n")

        # Find the last "Topologically Sorted" comment
        topo_match = None
        comment_block_lines: list[str] = []
        for i in range(len(lines) - 1, -1, -1):
            m = topo_pattern.search(lines[i])
            if m:
                topo_match = m
                # Collect comment lines below it until non-comment
                comment_block_lines = []
                for j in range(i + 1, len(lines)):
                    if lines[j].strip().startswith("#"):
                        comment_block_lines.append(lines[j])
                    else:
                        break
                break

        if not topo_match:
            continue

        source_nodes = [s.strip() for s in topo_match.group(1).split(",")]
        aten_ops = [s.strip() for s in topo_match.group(2).split(",")]

        # Parse source → aten mapping from comment block
        aten_to_source: dict[str, str] = {}
        for cline in comment_block_lines:
            sm = source_map_pattern.match(cline.strip())
            if sm:
                aten_to_source[sm.group(2)] = sm.group(1)

        kernel = TritonKernel(
            name=kname,
            source_code=kernel_sources.get(kname, ""),
            fused_aten_ops=aten_ops,
            fused_source_nodes=source_nodes,
            kernel_type="triton",
            aten_to_source_map=aten_to_source,
        )
        capture.kernels.append(kernel)


def _parse_source_mapping(comments: str) -> dict[str, str]:
    """Parse '# source_node => aten_node' mapping lines from comment block."""
    mapping: dict[str, str] = {}
    for line in comments.split("\n"):
        line = line.strip().lstrip("#").strip()
        if "=>" in line:
            parts = line.split("=>")
            if len(parts) == 2:
                src = parts[0].strip()
                aten_node = parts[1].strip()
                mapping[aten_node] = src
    return mapping


def _parse_call_method(code: str, capture: TritonCapture) -> None:
    """Parse the call() method to build execution-order call sequence."""
    call_match = re.search(r"def call\(self, args\):(.*?)(?=\nrunner\b|\nclass |\Z)", code, re.DOTALL)
    if not call_match:
        return

    call_body = call_match.group(1)
    lines = call_body.split("\n")
    current_source_nodes: list[str] = []
    current_aten_ops: list[str] = []
    known_triton_names = {k.name for k in capture.kernels}

    for line in lines:
        line = line.strip()

        comment_match = re.match(
            r"# Topologically Sorted Source Nodes: \[([^\]]*)\], Original ATen: \[([^\]]*)\]",
            line,
        )
        if comment_match:
            current_source_nodes = [s.strip() for s in comment_match.group(1).split(",")]
            current_aten_ops = [s.strip() for s in comment_match.group(2).split(",")]
            continue

        extern_match = re.match(r"extern_kernels\.(\w+)\(", line)
        if extern_match and current_aten_ops:
            fn_name = extern_match.group(1)
            extern_name = f"extern_kernels.{fn_name}"
            kernel = TritonKernel(
                name=extern_name,
                source_code="",
                fused_aten_ops=[f"aten.{fn_name}"],
                fused_source_nodes=current_source_nodes,
                kernel_type="extern",
            )
            capture.kernels.append(kernel)
            capture.call_sequence.append(
                {"name": extern_name, "type": "extern", "fn": fn_name}
            )
            current_source_nodes = []
            current_aten_ops = []
            continue

        triton_run = re.match(r"(\w+)\.run\(", line)
        if triton_run:
            name = triton_run.group(1)
            if name in known_triton_names:
                capture.call_sequence.append({"name": name, "type": "triton"})
            current_source_nodes = []
            current_aten_ops = []


@dataclass
class KernelMapping:
    """Maps FX graph node names to Triton/extern kernel names."""

    node_to_kernel: dict[str, str] = field(default_factory=dict)
    kernel_details: dict[str, TritonKernel] = field(default_factory=dict)
    kernel_call_order: list[str] = field(default_factory=list)
    unmapped_nodes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        n_mapped = len(self.node_to_kernel)
        n_kernels = len(self.kernel_details)
        n_unmapped = len(self.unmapped_nodes)
        return (
            f"KernelMapping: {n_mapped} nodes mapped to {n_kernels} kernels"
            f" ({n_unmapped} unmapped)"
        )


def _extract_short_op(target_str: str) -> str | None:
    """Extract short aten op name from a target string like 'torch.ops.aten.mm.default'."""
    if "aten." in target_str:
        parts = target_str.split(".")
        try:
            idx = parts.index("aten")
            return parts[idx + 1]
        except (ValueError, IndexError):
            pass
    if "getitem" in target_str:
        return "getitem"
    return None


def build_kernel_node_map(
    capture,
    triton_capture: TritonCapture,
) -> KernelMapping:
    """Build a mapping from FX graph node names to kernel names.

    Walks the inductor's execution-order call sequence and matches each
    kernel call to graph nodes by aten op type and topological position.
    Unmapped nodes (getitem, view, clone, etc.) are propagated from their
    nearest mapped producer or consumer.
    """
    mapping = KernelMapping()
    kernel_defs: dict[str, TritonKernel] = {}
    for k in triton_capture.kernels:
        if k.name not in kernel_defs:
            kernel_defs[k.name] = k
    mapping.kernel_details = kernel_defs
    mapping.kernel_call_order = [c["name"] for c in triton_capture.call_sequence]

    if not capture.forward_graphs:
        return mapping

    fg = capture.forward_graphs[0]

    # Build per-op-type queues in topological order
    op_queues: dict[str, list[str]] = {}
    all_compute: list[str] = []
    node_map: dict[str, Any] = {}
    for node in fg.graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        node_map[node.name] = node
        short_op = _extract_short_op(str(node.target))
        if short_op:
            op_queues.setdefault(short_op, []).append(node.name)
        all_compute.append(node.name)

    # Shape/bookkeeping ops: only assign via propagation, never direct match.
    # The inductor's fused_aten_ops includes internal lowered views/transposes
    # that don't correspond to our graph's shape ops.
    propagate_only = {
        "view", "t", "transpose", "reshape", "permute", "expand",
        "contiguous", "unsqueeze", "squeeze", "getitem",
        "clone", "detach", "split",
    }

    # Walk execution order, consume from queues
    for call in triton_capture.call_sequence:
        kname = call["name"]
        if call["type"] == "extern":
            fn = call["fn"]
            if fn in op_queues and op_queues[fn]:
                mapping.node_to_kernel[op_queues[fn].pop(0)] = kname
        else:
            kdef = kernel_defs.get(kname)
            if not kdef:
                continue
            for aten_op in kdef.fused_aten_ops:
                short_op = aten_op.replace("aten.", "").replace("prims.", "")
                if short_op in propagate_only:
                    continue
                if short_op in op_queues and op_queues[short_op]:
                    mapping.node_to_kernel[op_queues[short_op].pop(0)] = kname

    # Assign native CUDA ops (SDPA, etc.) their own group before propagation
    # so surrounding shape ops propagate to them rather than distant kernels.
    native_ops = {
        "_scaled_dot_product_efficient_attention": "cuda_sdpa",
        "_scaled_dot_product_flash_attention": "cuda_sdpa",
    }
    for node_name in all_compute:
        if node_name in mapping.node_to_kernel:
            continue
        node = node_map.get(node_name)
        if not node:
            continue
        short_op = _extract_short_op(str(node.target))
        group = native_ops.get(short_op) if short_op else None
        if group:
            mapping.node_to_kernel[node_name] = group
            if group not in mapping.kernel_details:
                mapping.kernel_details[group] = TritonKernel(
                    name=group,
                    source_code="",
                    fused_aten_ops=[f"aten.{short_op}"],
                    fused_source_nodes=[],
                    kernel_type="native",
                )

    # Propagation pass 1 (forward): inherit kernel from producer
    for node_name in all_compute:
        if node_name in mapping.node_to_kernel:
            continue
        node = node_map.get(node_name)
        if not node:
            continue
        for arg in node.args:
            if hasattr(arg, "name") and arg.name in mapping.node_to_kernel:
                mapping.node_to_kernel[node_name] = mapping.node_to_kernel[arg.name]
                break

    # Propagation pass 2 (backward): inherit kernel from consumer
    for node_name in reversed(all_compute):
        if node_name in mapping.node_to_kernel:
            continue
        node = node_map.get(node_name)
        if not node:
            continue
        for user in node.users:
            if hasattr(user, "name") and user.name in mapping.node_to_kernel:
                mapping.node_to_kernel[node_name] = mapping.node_to_kernel[user.name]
                break

    mapping.unmapped_nodes = [n for n in all_compute if n not in mapping.node_to_kernel]
    return mapping


def save_triton_kernels(
    capture: TritonCapture,
    output_dir: str,
    prefix: str = "",
) -> list[Path]:
    """Save all captured Triton kernel source files to a directory."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved = []

    for kernel in capture.kernels:
        if kernel.kernel_type != "triton" or not kernel.source_code:
            continue
        name = f"{prefix}{kernel.name}" if prefix else kernel.name
        path = out / f"{name}.py"

        header = f"# Fused aten ops: {', '.join(kernel.fused_aten_ops)}\n"
        header += f"# Source nodes: {', '.join(kernel.fused_source_nodes)}\n"
        if kernel.aten_to_source_map:
            header += "# Mapping:\n"
            for aten_name, src_name in kernel.aten_to_source_map.items():
                header += f"#   {src_name} => {aten_name}\n"
        header += "\n"

        path.write_text(header + kernel.source_code)
        saved.append(path)

    if capture.inductor_output_code:
        full_path = out / f"{prefix}full_output_code.py" if prefix else out / "full_output_code.py"
        full_path.write_text(capture.inductor_output_code)
        saved.append(full_path)

    return saved
