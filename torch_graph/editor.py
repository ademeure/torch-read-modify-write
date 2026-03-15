"""Edit FX graphs at the op level.

Supports inserting, removing, replacing, and rewiring individual nodes,
then recompiling into a runnable GraphModule.
"""

from __future__ import annotations

import copy
from collections import deque
from typing import Callable

import torch
from torch.fx import Graph, GraphModule, Node

from torch_graph.capture import CapturedGraph


class GraphEditor:
    """Mutable editor for FX graphs with undo support.

    Example:
        editor = GraphEditor(captured_graph)
        editor.replace_op("relu", torch.nn.functional.gelu)
        editor.remove_node("dropout")
        editor.insert_after("add", torch.mul, args_fn=lambda n: (n, 2.0))
        new_gm = editor.compile()
        output = new_gm(*inputs)
    """

    def __init__(self, source: CapturedGraph | GraphModule):
        if isinstance(source, CapturedGraph):
            self.original_gm = source.graph_module
            self.example_inputs = source.example_inputs
        else:
            self.original_gm = source
            self.example_inputs = None

        # Work on a deep copy
        self.gm = copy.deepcopy(self.original_gm)
        self.graph = self.gm.graph
        self._history: deque[GraphModule] = deque(maxlen=50)

    def _checkpoint(self) -> None:
        self._history.append(copy.deepcopy(self.gm))

    def undo(self) -> bool:
        """Undo the last edit."""
        if not self._history:
            return False
        self.gm = self._history.pop()
        self.graph = self.gm.graph
        return True

    def replace_op(
        self,
        node_name: str,
        new_target: Callable,
        new_args: tuple | None = None,
        new_kwargs: dict | None = None,
    ) -> Node:
        """Replace a node's target function (and optionally args/kwargs)."""
        self._checkpoint()
        node = _find_node(self.graph, node_name)
        node.target = new_target
        if new_args is not None:
            node.args = new_args
        if new_kwargs is not None:
            node.kwargs = new_kwargs
        self.graph.lint()
        return node

    def replace_all_ops(
        self,
        old_target: Callable,
        new_target: Callable,
    ) -> list[Node]:
        """Replace every occurrence of old_target with new_target."""
        self._checkpoint()
        replaced = [
            node for node in self.graph.nodes
            if node.target is old_target or (
                hasattr(node.target, "name") and hasattr(old_target, "name")
                and node.target.name() == old_target.name()
            )
        ]
        for node in replaced:
            node.target = new_target
        if replaced:
            self.graph.lint()
        return replaced

    def remove_node(self, node_name: str) -> bool:
        """Remove a node, rewiring its users to use its first input instead."""
        self._checkpoint()
        node = _find_node(self.graph, node_name)
        passthrough = next((a for a in node.args if isinstance(a, Node)), None)
        if passthrough is None and node.users:
            # Can't remove — no input to rewire to
            self._history.pop()
            return False
        if passthrough is not None:
            node.replace_all_uses_with(passthrough)
        self.graph.erase_node(node)
        self.graph.lint()
        return True

    def insert_after(
        self,
        node_name: str,
        target: Callable,
        args_fn: Callable[[Node], tuple] | None = None,
        kwargs: dict | None = None,
        name: str | None = None,
    ) -> Node:
        """Insert a new call_function node after the specified node.

        args_fn receives the reference node and returns the args tuple.
        Default: (reference_node,).
        """
        self._checkpoint()
        ref_node = _find_node(self.graph, node_name)
        args = args_fn(ref_node) if args_fn is not None else (ref_node,)
        with self.graph.inserting_after(ref_node):
            new_node = self.graph.call_function(
                target, args=args, kwargs=kwargs or {}
            )
            if name:
                new_node.name = name

        self.graph.lint()
        return new_node

    def insert_before(
        self,
        node_name: str,
        target: Callable,
        args_fn: Callable[[Node], tuple] | None = None,
        kwargs: dict | None = None,
        name: str | None = None,
    ) -> Node:
        """Insert a new call_function node before the specified node."""
        self._checkpoint()
        ref_node = _find_node(self.graph, node_name)
        if args_fn is not None:
            args = args_fn(ref_node)
        else:
            # Use the same first input as the reference node
            first = next((a for a in ref_node.args if isinstance(a, Node)), None)
            args = (first,) if first else ()
        with self.graph.inserting_before(ref_node):
            new_node = self.graph.call_function(
                target, args=args, kwargs=kwargs or {}
            )
            if name:
                new_node.name = name

        self.graph.lint()
        return new_node

    def rewire(self, node_name: str, input_idx: int, new_input_name: str) -> Node:
        """Change the input at index `input_idx` of a node to point to `new_input_name`."""
        self._checkpoint()
        node = _find_node(self.graph, node_name)
        new_input = _find_node(self.graph, new_input_name)
        args = list(node.args)
        args[input_idx] = new_input
        node.args = tuple(args)
        self.graph.lint()
        return node

    def fuse_nodes(
        self,
        node_names: list[str],
        fused_target: Callable,
        fused_name: str = "fused",
    ) -> Node:
        """Replace a group of nodes with a single fused operation.

        Collects all external Node inputs (args and kwargs) from every node in
        the group, deduplicates them, and passes them as positional args to
        fused_target. The fused node replaces the last node's outputs.

        If intermediate nodes have users outside the fused group, fused_target
        must return a tuple: (*intermediate_results, final_result), where
        intermediate_results are in node_names order (skipping nodes with no
        external users). getitem nodes are inserted to extract each result and
        rewire the external users.

        If no intermediate nodes have external users (common case), fused_target
        returns a single value as before.

        Non-Node args (scalars, dtypes, etc.) used by intermediate nodes are
        NOT forwarded — fused_target must hardcode those.
        """
        import operator

        self._checkpoint()
        nodes = [_find_node(self.graph, name) for name in node_names]
        last_node = nodes[-1]
        fused_set = set(node_names)

        # Find intermediate nodes whose results are needed outside the group
        exposed = [n for n in nodes[:-1]
                   if any(u.name not in fused_set for u in n.users)]

        # Collect external Node inputs from args AND kwargs of all fused nodes
        external_args = []
        for n in nodes:
            for arg in list(n.args) + list(n.kwargs.values()):
                if isinstance(arg, Node) and arg.name not in fused_set:
                    if arg not in external_args:
                        external_args.append(arg)

        with self.graph.inserting_after(last_node):
            fused_node = self.graph.call_function(
                fused_target, args=tuple(external_args)
            )
            fused_node.name = fused_name

        if exposed:
            # fused_target returns tuple: (*exposed_intermediates, final_result)
            # Create getitem nodes for each exposed intermediate
            insert_point = fused_node
            for i, intermediate in enumerate(exposed):
                with self.graph.inserting_after(insert_point):
                    gi = self.graph.call_function(
                        operator.getitem, args=(fused_node, i)
                    )
                    gi.name = f"{fused_name}_{intermediate.name}"
                intermediate.replace_all_uses_with(gi)
                insert_point = gi

            # Final result is the last tuple element
            with self.graph.inserting_after(insert_point):
                final_gi = self.graph.call_function(
                    operator.getitem, args=(fused_node, len(exposed))
                )
                final_gi.name = f"{fused_name}_out"
            last_node.replace_all_uses_with(final_gi)
        else:
            last_node.replace_all_uses_with(fused_node)

        # Remove fused nodes in reverse graph order (not node_names order)
        # to respect topological dependencies
        for n in reversed(list(self.graph.nodes)):
            if n.name in fused_set:
                self.graph.erase_node(n)

        self.graph.lint()
        return fused_node

    def add_logging(self, node_name: str) -> Node:
        """Insert a logging node after the specified node (for debugging)."""
        def _log_tensor(x, name=""):
            if isinstance(x, torch.Tensor):
                print(f"[GRAPH LOG] {name}: shape={x.shape}, dtype={x.dtype}, "
                      f"mean={x.float().mean():.4f}, std={x.float().std():.4f}")
            return x

        return self.insert_after(
            node_name,
            _log_tensor,
            args_fn=lambda n: (n, node_name),
            name=f"log_{node_name}",
        )

    def compile(self) -> GraphModule:
        """Recompile the edited graph into an executable GraphModule.

        Returns self.gm directly — deep-copy the result if you need an
        independent snapshot.
        """
        self.graph.lint()
        self.gm.recompile()
        return self.gm

    def validate(self) -> bool:
        """Validate the graph can be compiled and (if inputs available) executed.

        Works on a deep copy so self.gm is never mutated.
        """
        try:
            test_gm = copy.deepcopy(self.gm)
            test_gm.graph.lint()
            test_gm.recompile()
            if self.example_inputs:
                test_gm(*self.example_inputs)
            return True
        except Exception as e:
            print(f"Validation failed: {e}")
            return False

    def diff(self) -> str:
        """Show a diff between the original and edited graph code."""
        if self.original_gm.code == self.gm.code:
            return "(no changes)"
        import difflib
        lines = difflib.unified_diff(
            self.original_gm.code.strip().split("\n"),
            self.gm.code.strip().split("\n"),
            lineterm="",
        )
        return "\n".join(["--- Original", "+++ Edited", ""] + list(lines))


def _find_node(graph: Graph, name: str) -> Node:
    for node in graph.nodes:
        if node.name == name:
            return node
    raise KeyError(
        f"Node '{name}' not found. Available: {[n.name for n in graph.nodes]}"
    )
