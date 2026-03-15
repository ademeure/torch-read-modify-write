"""Shared canonical IR for FX/AOT graphs.

This module is the semantic source of truth for graph serialization and Python
code generation. Viewer JSON should remain a separate, intentionally lossy
projection; executable Python and lossless IR JSON both derive from here.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.fx import GraphModule, Node


def target_to_str(target: Any) -> str:
    """Normalize a node target to a stable string."""
    if hasattr(target, "__name__"):
        return target.__name__
    if hasattr(target, "name"):
        try:
            return target.name()
        except Exception:
            pass
    return str(target)


def tensor_meta(val: Any) -> dict[str, Any] | None:
    """Extract compact tensor metadata when available."""
    if isinstance(val, torch.Tensor):
        return {
            "shape": list(val.shape),
            "dtype": str(val.dtype),
            "device": str(val.device),
        }
    return None


# Plumbing ops — reshape/index/alias without real compute.  Mirrors op_dump._VIEW_TARGETS.
_VIEW_TARGETS = frozenset({
    "view", "reshape", "permute", "transpose", "expand", "contiguous",
    "unsqueeze", "squeeze", "slice", "select", "split", "unbind", "chunk",
    "narrow", "as_strided", "alias", "detach", "clone", "t", "flatten",
    "getitem",
})


def _is_view_node(node: Node) -> bool:
    if node.op not in ("call_function", "call_method"):
        return False
    # callable_to_str → "aten.t", "aten.squeeze.dim", "operator.getitem"
    parts = callable_to_str(node.target).lower().split(".")
    op_name = parts[1] if len(parts) >= 2 and parts[0] == "aten" else parts[-1]
    return op_name in _VIEW_TARGETS


def _stride_comment(node: Node) -> str:
    """Trailing comment: strides, contiguity, and view flag for output tensors."""
    val = node.meta.get("val")
    view = _is_view_node(node)

    def _fmt(t: torch.Tensor) -> str:
        return f"strides={tuple(t.stride())}, contiguous={t.is_contiguous()}"

    if isinstance(val, torch.Tensor):
        return f"  # {_fmt(val)}, view={view}"
    if isinstance(val, (tuple, list)):
        parts = [f"out{i}: {_fmt(v)}" for i, v in enumerate(val) if isinstance(v, torch.Tensor)]
        if parts:
            return "  # " + "; ".join(parts) + f", view={view}"
    return f"  # view={view}" if view else ""


def placeholder_annotation(node: Node) -> str:
    """Return the exported placeholder type annotation string."""
    meta = tensor_meta(node.meta.get("val"))
    if not meta:
        return ""
    dims = ", ".join(str(s) for s in meta["shape"])
    dtype_name = meta["dtype"].split(".")[-1]
    return f"{dtype_name}[{dims}]"


def value_to_ir(value: Any) -> dict[str, Any]:
    """Serialize a node arg/kwarg/return value."""
    if isinstance(value, Node):
        return {"node": value.name}
    if value is None:
        return {"kind": "none", "value": None}
    if isinstance(value, bool):
        return {"kind": "bool", "value": value}
    if isinstance(value, int):
        return {"kind": "int", "value": value}
    if isinstance(value, float):
        if value == float("inf"):
            return {"kind": "float", "value": "inf"}
        if value == float("-inf"):
            return {"kind": "float", "value": "-inf"}
        return {"kind": "float", "value": value}
    if isinstance(value, str):
        return {"kind": "str", "value": value}
    if isinstance(value, slice):
        return {
            "kind": "slice",
            "start": value_to_ir(value.start),
            "stop": value_to_ir(value.stop),
            "step": value_to_ir(value.step),
        }
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [value_to_ir(v) for v in value]}
    if isinstance(value, list):
        return {"kind": "list", "items": [value_to_ir(v) for v in value]}
    if isinstance(value, dict):
        return {
            "kind": "dict",
            "items": [
                {"key": value_to_ir(k), "value": value_to_ir(v)}
                for k, v in value.items()
            ],
        }
    if isinstance(value, torch.dtype):
        return {"kind": "torch_dtype", "value": str(value)}
    if isinstance(value, torch.device):
        return {"kind": "torch_device", "value": str(value)}
    if isinstance(value, torch.layout):
        return {"kind": "torch_layout", "value": str(value)}
    if isinstance(value, torch.memory_format):
        return {"kind": "torch_memory_format", "value": str(value)}
    if isinstance(value, torch.Tensor):
        return {
            "kind": "tensor_literal",
            "meta": tensor_meta(value),
            "python": repr(value),
        }
    return {
        "kind": "python_repr",
        "python": repr(value),
    }


def ir_value_to_python(value: dict[str, Any]) -> str:
    """Convert an IR value back to Python source."""
    if "node" in value:
        return value["node"]
    kind = value["kind"]
    if kind == "none":
        return "None"
    if kind in {"bool", "int", "str"}:
        return repr(value["value"])
    if kind == "float":
        if value["value"] == "inf":
            return 'float("inf")'
        if value["value"] == "-inf":
            return 'float("-inf")'
        return repr(value["value"])
    if kind == "slice":
        start = ir_value_to_python(value["start"])
        stop = ir_value_to_python(value["stop"])
        step = ir_value_to_python(value["step"])
        return f"slice({start}, {stop}, {step})"
    if kind == "tuple":
        items = ", ".join(ir_value_to_python(v) for v in value["items"])
        if len(value["items"]) == 1:
            return f"({items},)"
        return f"({items})"
    if kind == "list":
        return "[" + ", ".join(ir_value_to_python(v) for v in value["items"]) + "]"
    if kind == "dict":
        return "{" + ", ".join(
            f"{ir_value_to_python(item['key'])}: {ir_value_to_python(item['value'])}"
            for item in value["items"]
        ) + "}"
    if kind == "torch_dtype":
        return value["value"]
    if kind == "torch_device":
        return f"torch.device({value['value']!r})"
    if kind in {"torch_layout", "torch_memory_format"}:
        return value["value"]
    if kind in {"python_repr", "tensor_literal"}:
        return value["python"]
    raise ValueError(f"Unsupported IR value kind: {kind}")


def graph_output_ir(gm: GraphModule) -> list[dict[str, Any]]:
    """Serialize the output expression in graph order."""
    out_node = next((n for n in gm.graph.nodes if n.op == "output"), None)
    if out_node is None or not out_node.args:
        return []
    out = out_node.args[0]
    if isinstance(out, (tuple, list)):
        return [value_to_ir(v) for v in out]
    return [value_to_ir(out)]


def tensor_to_constructor(
    tensor: torch.Tensor,
    name: str,
    threshold: int = 1000,
    comment: str = "",
) -> str:
    """Convert a tensor to a Python constructor string."""
    from torch_graph._utils import is_fake

    suffix = f"  # {comment}" if comment else ""

    def _device_kwarg(t: torch.Tensor) -> str:
        """Return device= kwarg string for non-CPU tensors."""
        if str(t.device) != "cpu":
            return f", device='{t.device}'"
        return ""

    if is_fake(tensor):
        shape = list(tensor.shape)
        ctor = "torch.randn" if tensor.dtype.is_floating_point else "torch.zeros"
        dev = _device_kwarg(tensor)
        return f"{name} = {ctor}({shape}, dtype={tensor.dtype}{dev}){suffix or f'  # shape={shape}'}"

    try:
        _ = tensor.untyped_storage()
    except Exception:
        shape = list(tensor.shape)
        ctor = "torch.randn" if tensor.dtype.is_floating_point else "torch.zeros"
        dev = _device_kwarg(tensor)
        return f"{name} = {ctor}({shape}, dtype={tensor.dtype}{dev}){suffix or f'  # shape={shape}'}"

    if tensor.numel() <= threshold:
        try:
            dev = _device_kwarg(tensor)
            if tensor.ndim == 0:
                return f"{name} = torch.tensor({tensor.item()}, dtype={tensor.dtype}{dev}){suffix}"
            data = tensor.detach().cpu().tolist()
            return f"{name} = torch.tensor({data}, dtype={tensor.dtype}{dev}){suffix}"
        except Exception:
            shape = list(tensor.shape)
            ctor = "torch.randn" if tensor.dtype.is_floating_point else "torch.zeros"
            dev = _device_kwarg(tensor)
            return f"{name} = {ctor}({shape}, dtype={tensor.dtype}{dev}){suffix or f'  # shape={shape}'}"

    shape = list(tensor.shape)
    return f'{name} = weights["{name}"]{suffix or f"  # shape={shape}, dtype={tensor.dtype}"}'


def callable_to_str(target: Any) -> str:
    """Convert a callable target to its fully-qualified string."""
    if hasattr(target, "__module__") and hasattr(target, "__qualname__"):
        module = target.__module__
        qualname = target.__qualname__

        if "torch.ops" in str(target) or (
            hasattr(target, "name") and callable(getattr(target, "name", None))
        ):
            try:
                name = target.name()
                if "::" in name:
                    namespace, remainder = name.split("::", 1)
                    op_parts = remainder.split(".")
                    prefix = f"{namespace}." if namespace == "aten" else f"torch.ops.{namespace}."
                    if len(op_parts) == 1:
                        return f"{prefix}{op_parts[0]}"
                    if op_parts[-1] == "default":
                        return f"{prefix}{'.'.join(op_parts[:-1])}"
                    return f"{prefix}{'.'.join(op_parts)}"
                return f"torch.ops.{name}"
            except Exception:
                pass

        if module == "_operator":
            return f"operator.{qualname}"
        if module == "torch":
            return f"torch.{qualname}"
        if module and module.startswith("torch."):
            return f"{module}.{qualname}"

    text = str(target)
    if "aten." in text:
        return f"torch.ops.{text.split('torch.ops.')[-1]}" if "torch.ops." in text else text
    return text


def _is_triton_wrapper(node: Node) -> bool:
    """Check if a node is a triton_kernel_wrapper_functional call."""
    return (node.op == "call_function"
            and getattr(node.target, "__name__", "") == "triton_kernel_wrapper_functional")


def _triton_node_to_python(node: Node) -> str:
    """Convert a triton_kernel_wrapper_functional node to Python.

    Emits:
      1. Clone statements for output tensors (tensors_to_clone)
      2. TMA descriptor creation for args with tma_descriptor_metadata
      3. A direct Triton kernel launch: kernel_name[grid](**tensor_args, **const_args)
      4. A dict assignment so getitem nodes can resolve outputs

    The kernel function name is looked up from the kernel_side_table.
    """
    from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table

    kernel_idx = node.kwargs.get("kernel_idx", 0)
    constant_args_idx = node.kwargs.get("constant_args_idx", 0)
    grid = node.kwargs.get("grid", [(1, 1, 1)])
    tensor_kwargs = node.kwargs.get("kwargs", {})
    tensors_to_clone = node.kwargs.get("tensors_to_clone", [])
    tma_metadata = node.kwargs.get("tma_descriptor_metadata", {})

    # Look up kernel name
    try:
        kernel_obj = kernel_side_table.get_kernel(kernel_idx)
        kernel_name = getattr(kernel_obj.fn, "__name__", f"_triton_kernel_{kernel_idx}")
    except Exception:
        kernel_name = f"_triton_kernel_{kernel_idx}"

    # Look up constant args
    try:
        const_args = kernel_side_table.get_constant_args(constant_args_idx)
    except Exception:
        const_args = {}

    lines = []

    # Clone output tensors
    clone_map = {}
    for arg_name in tensors_to_clone:
        tensor_node = tensor_kwargs.get(arg_name)
        if tensor_node is not None:
            src = tensor_node.name if isinstance(tensor_node, Node) else str(tensor_node)
            clone_name = f"_{node.name}_{arg_name}"
            lines.append(f"{clone_name} = {src}.clone()")
            clone_map[arg_name] = clone_name

    # Create TMA descriptors for args that need them
    tma_var_map = {}  # arg_name -> TMA variable name
    for arg_name, meta in tma_metadata.items():
        # meta format: ('stable', (block_shape,))
        if isinstance(meta, tuple) and len(meta) >= 2:
            block_shape = meta[1]
            if isinstance(block_shape, tuple) and len(block_shape) >= 1:
                block_shape = list(block_shape[0])
            else:
                block_shape = list(block_shape)
        else:
            continue

        # Get the tensor source (use clone if it was cloned)
        if arg_name in clone_map:
            tensor_src = clone_map[arg_name]
        else:
            tensor_node = tensor_kwargs.get(arg_name)
            if isinstance(tensor_node, Node):
                tensor_src = tensor_node.name
            else:
                continue

        tma_var = f"_{node.name}_{arg_name}_tma"
        lines.append(f"{tma_var} = TensorDescriptor.from_tensor({tensor_src}, {block_shape})")
        tma_var_map[arg_name] = tma_var

    # Build kernel call arguments
    call_args = []
    for arg_name, tensor_node in tensor_kwargs.items():
        if arg_name in tma_var_map:
            call_args.append(f"{arg_name}={tma_var_map[arg_name]}")
        elif arg_name in clone_map:
            call_args.append(f"{arg_name}={clone_map[arg_name]}")
        elif isinstance(tensor_node, Node):
            call_args.append(f"{arg_name}={tensor_node.name}")
        else:
            call_args.append(f"{arg_name}={ir_value_to_python(value_to_ir(tensor_node))}")

    for arg_name, val in const_args.items():
        call_args.append(f"{arg_name}={val!r}")

    # Format grid
    if len(grid) == 1 and isinstance(grid[0], (tuple, list)):
        grid_str = repr(tuple(grid[0]))
    else:
        grid_str = repr(tuple(grid))

    lines.append(f"{kernel_name}[{grid_str}]({', '.join(call_args)})")

    # Build result dict (for getitem nodes to resolve)
    # Use the original tensor names (not TMA descriptors) for dict values
    dict_items = []
    for arg_name, tensor_node in tensor_kwargs.items():
        if arg_name in clone_map:
            dict_items.append(f"'{arg_name}': {clone_map[arg_name]}")
        elif isinstance(tensor_node, Node):
            dict_items.append(f"'{arg_name}': {tensor_node.name}")
    lines.append(f"{node.name} = {{{', '.join(dict_items)}}}")

    return "\n".join(lines)


def fx_node_to_python(node: Node, graph_module: GraphModule | None = None) -> str:
    """Convert a single FX node to a Python statement."""
    if node.op == "placeholder":
        return ""

    if node.op == "output":
        if not node.args:
            return "return None"
        out = node.args[0]
        if isinstance(out, (tuple, list)):
            return ir_return_to_python([value_to_ir(v) for v in out])
        return ir_return_to_python([value_to_ir(out)])

    if node.op == "get_attr":
        # Try to get the real tensor from the GraphModule first
        val = None
        if graph_module is not None:
            try:
                parts = node.target.split(".")
                obj = graph_module
                for p in parts:
                    obj = getattr(obj, p)
                val = obj
            except Exception:
                pass
        if val is None:
            val = node.meta.get("val")
        if isinstance(val, torch.Tensor):
            return tensor_to_constructor(val, node.name)
        return f"{node.name} = torch.tensor(0.0)  # get_attr: {node.target}"

    # Handle Triton kernel wrapper nodes specially
    if _is_triton_wrapper(node):
        return _triton_node_to_python(node)

    target_str = callable_to_str(node.target) if node.op == "call_function" else str(node.target)
    args_strs = [ir_value_to_python(value_to_ir(arg)) for arg in node.args]
    kwargs_strs = [f"{k}={ir_value_to_python(value_to_ir(v))}" for k, v in node.kwargs.items()]
    all_args = ", ".join(args_strs + kwargs_strs)

    val = node.meta.get("val")
    meta = tensor_meta(val)
    type_annotation = ""
    if meta:
        type_annotation = f": '{meta['dtype'].split('.')[-1]}[{', '.join(str(s) for s in meta['shape'])}]'"

    if node.op == "call_method":
        if args_strs:
            obj = args_strs[0]
            rest = ", ".join(args_strs[1:] + kwargs_strs)
            return f"{node.name}{type_annotation} = {obj}.{target_str}({rest})"
        return f"{node.name}{type_annotation} = {target_str}({all_args})"

    return f"{node.name}{type_annotation} = {target_str}({all_args})"


def graph_to_ir(
    graph_module: GraphModule,
    *,
    fn_name: str = "forward",
    placeholder_display_names: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Serialize a graph as the canonical internal IR."""
    placeholders = []
    nodes = []
    for node in graph_module.graph.nodes:
        if node.op == "placeholder":
            entry = {
                "name": node.name,
                "fx_op": node.op,
                "target": target_to_str(node.target),
                "annotation": placeholder_annotation(node),
                "meta": tensor_meta(node.meta.get("val")) or {},
            }
            if placeholder_display_names and node.name in placeholder_display_names:
                entry["display_name"] = placeholder_display_names[node.name]
            placeholders.append(entry)
            continue
        if node.op == "output":
            continue

        nodes.append(
            {
                "name": node.name,
                "fx_op": node.op,
                "target": target_to_str(node.target),
                "args": [value_to_ir(arg) for arg in node.args],
                "kwargs": {k: value_to_ir(v) for k, v in node.kwargs.items()},
                "meta": tensor_meta(node.meta.get("val")) or {},
                "python": fx_node_to_python(node, graph_module),
                "stride_comment": _stride_comment(node),
            }
        )

    return {
        "fn_name": fn_name,
        "placeholders": placeholders,
        "nodes": nodes,
        "returns": graph_output_ir(graph_module),
    }


def ir_return_to_python(returns: list[dict[str, Any]]) -> str:
    """Convert serialized graph returns back to a Python return statement."""
    if not returns:
        return "return None"
    if len(returns) == 1:
        return f"return {ir_value_to_python(returns[0])}"
    return "return (" + ", ".join(ir_value_to_python(v) for v in returns) + ",)"


def ir_graph_to_python(ir_graph: dict[str, Any]) -> str:
    """Rebuild a standalone Python function from an IR graph dict."""
    lines = []
    fn_name = ir_graph.get("fn_name", "forward")
    placeholders = ir_graph.get("placeholders", [])
    if len(placeholders) <= 3:
        params = ", ".join(
            f"{p['name']}: '{p['annotation']}'" if p.get("annotation") else p["name"]
            for p in placeholders
        )
        lines.append(f"def {fn_name}({params}):")
    else:
        lines.append(f"def {fn_name}(")
        for placeholder in placeholders:
            annotation = f": '{placeholder['annotation']}'" if placeholder.get("annotation") else ""
            lines.append(f"    {placeholder['name']}{annotation},")
        lines.append("):")

    for node in ir_graph.get("nodes", []):
        for sub in node["python"].split("\n"):
            lines.append(f"    {sub}")
    lines.append(f"    {ir_return_to_python(ir_graph.get('returns', []))}")
    return "\n".join(lines) + "\n"
