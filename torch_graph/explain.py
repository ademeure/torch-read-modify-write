"""One-liner model explanation: capture, inspect, and optionally verify/profile.

Usage::

    from torch_graph import explain
    result = explain(model, x)
    print(result)
"""

from __future__ import annotations

import io
import time
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from torch_graph.export import AtenCapture, capture_aten_graphs
from torch_graph.inspector import GraphInspector


@dataclass
class ExplainResult:
    """Structured result from ``explain()``."""

    model_name: str
    num_parameters: int
    num_buffers: int
    param_memory_mb: float
    num_forward_ops: int
    num_backward_ops: int
    op_counts: dict[str, int]
    op_categories: dict[str, list[str]]
    shapes: list[dict[str, Any]]
    capture: AtenCapture
    verification: dict[str, Any] | None = None
    profile_data: dict[str, Any] | None = None
    capture_time_s: float = 0.0

    def summary(self) -> str:
        """Formatted text summary."""
        lines = []
        lines.append(f"=== {self.model_name} ===")
        lines.append(f"Parameters: {self.num_parameters:,} ({self.param_memory_mb:.1f} MB)")
        lines.append(f"Buffers: {self.num_buffers}")
        lines.append(f"Forward ops: {self.num_forward_ops}")
        lines.append(f"Backward ops: {self.num_backward_ops}")
        lines.append(f"Capture time: {self.capture_time_s:.2f}s")

        # Top ops
        lines.append("")
        lines.append("Top ops:")
        for op, count in list(self.op_counts.items())[:10]:
            lines.append(f"  {op}: {count}")

        # Categories
        lines.append("")
        lines.append("Op categories:")
        for cat, ops in self.op_categories.items():
            lines.append(f"  {cat}: {len(ops)}")

        # Verification
        if self.verification is not None:
            lines.append("")
            v = self.verification
            fwd = v.get("forward")
            if fwd:
                lines.append(f"Verification forward: {'PASS' if fwd.get('matches') else 'FAIL'}")
            bwd = v.get("backward")
            if bwd:
                lines.append(f"Verification backward: {'PASS' if bwd.get('matches') else 'FAIL'}")

        # Profile
        if self.profile_data is not None:
            lines.append("")
            pd = self.profile_data
            if "peak_memory_mb" in pd:
                lines.append(f"Peak CUDA memory: {pd['peak_memory_mb']:.1f} MB")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def explain(
    model: nn.Module,
    *args,
    verify: bool = False,
    profile: bool = False,
    dynamic: bool = False,
    verbose: bool = True,
    **kwargs,
) -> ExplainResult:
    """Capture and explain a model in one call.

    Args:
        model: PyTorch model to explain.
        *args: Sample inputs for the model.
        verify: If True, run verification against eager execution.
        profile: If True, collect CUDA profiling data.
        dynamic: If True, capture with dynamic shapes.
        verbose: If True, print the summary to stdout.
        **kwargs: Extra keyword arguments forwarded to the model.
    """
    model_name = type(model).__name__

    # Count params/buffers
    num_params = sum(p.numel() for p in model.parameters())
    num_buffers = sum(1 for _ in model.buffers())
    param_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)

    # Capture
    t0 = time.monotonic()
    output, capture = capture_aten_graphs(model, *args, dynamic=dynamic, **kwargs)
    capture_time = time.monotonic() - t0

    # Inspect forward
    fw_inspector = GraphInspector(capture.forward_graphs[0]) if capture.forward_graphs else None
    fw_op_counts = fw_inspector.op_counts() if fw_inspector else {}
    fw_categories = fw_inspector.op_categories() if fw_inspector else {}
    fw_shapes = fw_inspector.shapes_table() if fw_inspector else []
    num_fw_ops = sum(fw_op_counts.values())

    # Inspect backward
    bw_inspector = GraphInspector(capture.backward_graphs[0]) if capture.backward_graphs else None
    num_bw_ops = sum(bw_inspector.op_counts().values()) if bw_inspector else 0

    # Verification
    verification = None
    if verify:
        from torch_graph.tensor_dump import verify_against_model
        verification = verify_against_model(model, *args, verbose=False, **kwargs)

    # Profile
    profile_data = None
    if profile and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            model(*args, **kwargs)
        peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
        profile_data = {"peak_memory_mb": peak_mem}

    result = ExplainResult(
        model_name=model_name,
        num_parameters=num_params,
        num_buffers=num_buffers,
        param_memory_mb=param_mb,
        num_forward_ops=num_fw_ops,
        num_backward_ops=num_bw_ops,
        op_counts=fw_op_counts,
        op_categories=fw_categories,
        shapes=fw_shapes,
        capture=capture,
        verification=verification,
        profile_data=profile_data,
        capture_time_s=capture_time,
    )

    if verbose:
        print(result.summary())

    return result
