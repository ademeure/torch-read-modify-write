"""
Graph capture via custom TorchDynamo backend.

Intercepts torch.compile to extract FX GraphModules before they reach
the real backend (inductor, etc.). This gives you FX-level graphs with
module structure preserved — useful for visualization and editing.

For aten-level decomposition (forward + backward as explicit ops),
use export.py's capture_aten_graphs() instead.

Usage:
    capture = GraphCapture()

    @torch.compile(backend=capture.backend)
    def fn(x):
        return x + 1

    fn(torch.randn(10))
    for g in capture.graphs:
        print(g.readable)

    # Or one-shot:
    result, capture = capture_graphs(my_model, input_tensor)
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import torch
from torch.fx import Graph, GraphModule

logger = logging.getLogger("torch_graph")

# -----------------------------------------------------------------------------
# A single captured FX graph snapshot

@dataclass
class CapturedGraph:
    """Snapshot of one FX graph from TorchDynamo compilation."""
    graph_module: GraphModule
    example_inputs: list[torch.Tensor]
    graph_id: int

    # Convenience accessors — thin wrappers so callers don't need to know
    # about the GraphModule internals
    @property
    def graph(self) -> Graph: return self.graph_module.graph
    @property
    def readable(self) -> str: return self.graph_module.print_readable(print_output=False)
    @property
    def code(self) -> str: return self.graph_module.code
    @property
    def num_nodes(self) -> int: return len(list(self.graph.nodes))

    @property
    def num_ops(self) -> int:
        """Count of actual compute ops (excludes placeholders, output, get_attr)."""
        return sum(1 for n in self.graph.nodes
                   if n.op in ("call_function", "call_method", "call_module"))

    def __repr__(self) -> str:
        return (f"CapturedGraph(id={self.graph_id}, nodes={self.num_nodes}, "
                f"ops={self.num_ops}, inputs={len(self.example_inputs)})")

# -----------------------------------------------------------------------------
# The capture engine

class GraphCapture:
    """Accumulates all FX graphs produced by TorchDynamo during torch.compile.

    Pass capture.backend as the backend= argument to torch.compile. Each time
    Dynamo compiles a subgraph, we deep-copy it (so later mutations don't
    affect our snapshot) and optionally forward to a real backend.

    Supports iteration: len(capture), capture[0], for g in capture.
    """

    def __init__(self, passthrough_backend: str | Callable | None = None, deep_copy: bool = True):
        self.graphs: list[CapturedGraph] = []
        self._counter = 0
        self._deep_copy = deep_copy
        self._passthrough_backend = passthrough_backend

    def backend(self, gm: GraphModule, example_inputs: list[torch.Tensor]) -> Any:
        """TorchDynamo backend entry point — called once per compiled subgraph."""
        graph_module = copy.deepcopy(gm) if self._deep_copy else gm
        inputs_copy = [inp.clone().detach() for inp in example_inputs]
        self.graphs.append(CapturedGraph(
            graph_module=graph_module, example_inputs=inputs_copy, graph_id=self._counter,
        ))
        self._counter += 1
        logger.debug(f"Captured graph #{self._counter - 1}: "
                     f"{self.graphs[-1].num_nodes} nodes, {self.graphs[-1].num_ops} ops")

        # Forward to real backend if specified (e.g. "inductor" to still get fast code)
        if self._passthrough_backend is not None:
            if isinstance(self._passthrough_backend, str):
                from torch._dynamo.backends.registry import lookup_backend
                return lookup_backend(self._passthrough_backend)(gm, example_inputs)
            return self._passthrough_backend(gm, example_inputs)
        return gm.forward

    def clear(self):
        self.graphs.clear()
        self._counter = 0

    def summary(self) -> str:
        lines = [f"Captured {len(self.graphs)} graph(s):"]
        for g in self.graphs:
            lines.append(f"  [{g.graph_id}] {g.num_nodes} nodes, {g.num_ops} ops")
        return "\n".join(lines)

    def __len__(self) -> int: return len(self.graphs)
    def __getitem__(self, idx: int) -> CapturedGraph: return self.graphs[idx]
    def __iter__(self) -> Iterator[CapturedGraph]: return iter(self.graphs)

# -----------------------------------------------------------------------------
# One-shot convenience function

def capture_graphs(
    fn: Callable, *args,
    passthrough_backend: str | Callable | None = None,
    **kwargs,
) -> tuple[Any, GraphCapture]:
    """Compile and run fn in one shot, return (result, capture).

    Resets the compiler state first so we get a clean capture.
    If you need to capture multiple calls, use GraphCapture directly.
    """
    capture = GraphCapture(passthrough_backend=passthrough_backend)
    torch.compiler.reset()
    compiled_fn = torch.compile(fn, backend=capture.backend)
    result = compiled_fn(*args, **kwargs)
    return result, capture
