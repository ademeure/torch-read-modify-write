"""Compile and cache inline CUDA kernels for use in modified aten .py files.

Usage in a modified aten .py file::

    from torch_graph.cuda_inline import load_cuda

    _MY_KERNEL_CUDA = r'''
    #include <torch/extension.h>

    torch::Tensor my_fused_op(torch::Tensor x, torch::Tensor weight) {
        // Use ATen C++ ops or raw CUDA kernels
        return torch::mm(x, weight.t());
    }
    '''

    _my_kernel = None

    def my_layer(x, weight):
        global _my_kernel
        if _my_kernel is None:
            _my_kernel = load_cuda("my_kernel", _MY_KERNEL_CUDA, ["my_fused_op"])
        return (_my_kernel.my_fused_op(x, weight),)

The kernel compiles on first call (~10-30s) and is cached on disk for
subsequent runs.  Changing the CUDA source triggers a one-time recompile.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import logging
import os
import re
import sys
import textwrap
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

# In-memory cache: content-hash key → loaded module
_module_cache: dict[str, object] = {}


def _extract_forward_declarations(cuda_source: str, function_names: list[str]) -> str:
    """Extract C++ forward declarations for exported functions from CUDA source.

    ``load_inline()`` generates a ``main.cpp`` with pybind11 bindings that
    reference the exported functions.  When those functions live in a ``.cu``
    file, the compiler cannot see them without forward declarations in a
    ``.cpp`` translation unit.  This helper parses the CUDA source to find
    the function signatures and produces the required declarations.
    """
    declarations = []
    for name in function_names:
        # Match: <return_type> <name>(<params>) {
        # Return type can be multi-word: torch::Tensor, std::tuple<...>, void, etc.
        # We look for the function name followed by ( and match back to find the return type.
        pattern = (
            r'^[ \t]*'                      # leading whitespace
            r'([\w:<>, \t]+?)'              # return type (non-greedy)
            r'\s+' + re.escape(name) +      # function name
            r'\s*\(([^)]*)\)'               # parameter list
            r'\s*\{'                         # opening brace
        )
        match = re.search(pattern, cuda_source, re.MULTILINE)
        if match:
            ret_type = match.group(1).strip()
            params = match.group(2).strip()
            declarations.append(f"{ret_type} {name}({params});")

    return "\n".join(declarations)


def load_cuda(
    name: str,
    cuda_source: str,
    functions: list[str],
    cpp_source: str = "",
    extra_cuda_cflags: list[str] | None = None,
    extra_cflags: list[str] | None = None,
    verbose: bool = False,
) -> object:
    """Compile and cache a CUDA kernel from inline source strings.

    The compiled shared library is cached on disk under
    ``~/.cache/torch_extensions/``.  A content hash of the source is included
    in the module name so that edits trigger a recompile while unchanged
    sources are loaded instantly.

    Forward declarations for the exported *functions* are auto-extracted from
    *cuda_source* so callers don't need to duplicate signatures in
    *cpp_source*.

    Args:
        name: Human-readable base name (e.g. ``"residual_block"``).
        cuda_source: CUDA C++ source code.  Must define the functions listed
            in *functions* and include ``<torch/extension.h>``.
        functions: C++ function names to export to Python.
        cpp_source: Optional additional C++ source (header declarations etc.).
            If empty, forward declarations are auto-generated from
            *cuda_source*.
        extra_cuda_cflags: Extra flags for ``nvcc`` (default ``["-O3"]``).
        extra_cflags: Extra flags for the host C++ compiler.
        verbose: If True, print compilation commands.

    Returns:
        A Python module with the requested *functions* as callable attributes.
    """
    from torch.utils.cpp_extension import load_inline

    # Auto-generate forward declarations if caller didn't provide cpp_source
    if not cpp_source.strip():
        cpp_source = _extract_forward_declarations(cuda_source, functions)

    # Content hash → deterministic module name (no recompile if source unchanged)
    h = hashlib.sha256((cuda_source + cpp_source).encode()).hexdigest()[:16]
    key = f"{name}_{h}"

    if key in _module_cache:
        return _module_cache[key]

    if extra_cuda_cflags is None:
        extra_cuda_cflags = ["-O3"]

    logger.info("Compiling CUDA kernel '%s' (hash %s) ...", name, h)

    cpp_sources = [cpp_source] if cpp_source else []

    mod = load_inline(
        name=key,
        cpp_sources=cpp_sources,
        cuda_sources=[cuda_source],
        functions=functions,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_cflags=extra_cflags or [],
        verbose=verbose,
    )

    _module_cache[key] = mod
    logger.info("CUDA kernel '%s' compiled and cached.", name)
    return mod


def load_cuda_precompiled(so_path: str | Path) -> object:
    """Load a pre-compiled CUDA extension from a shared library.

    Use this when you have already compiled a ``.so`` file (e.g. via
    ``load_cuda`` on a build machine) and want zero-overhead loading
    without invoking the compiler at all.

    Args:
        so_path: Path to the compiled ``.so`` file.

    Returns:
        A Python module with the exported functions as callable attributes.
    """
    so_path = Path(so_path)
    if not so_path.exists():
        raise FileNotFoundError(f"Pre-compiled CUDA extension not found: {so_path}")

    module_name = so_path.stem
    if module_name in _module_cache:
        return _module_cache[module_name]

    spec = importlib.util.spec_from_file_location(module_name, str(so_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {so_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    _module_cache[module_name] = mod
    return mod


def clear_cache() -> None:
    """Clear the in-memory module cache (useful for testing)."""
    _module_cache.clear()


# ---------------------------------------------------------------------------
# Template generation helpers
# ---------------------------------------------------------------------------


def _torch_dtype_to_cpp(dtype_str: str) -> str:
    """Map a torch dtype string to its C++ scalar type."""
    mapping = {
        "float32": "float",
        "float64": "double",
        "float16": "at::Half",
        "bfloat16": "at::BFloat16",
        "int32": "int",
        "int64": "int64_t",
        "int16": "int16_t",
        "int8": "int8_t",
        "uint8": "uint8_t",
        "bool": "bool",
    }
    return mapping.get(dtype_str, "float")


def _parse_annotation(ann: str) -> tuple[str, list[str]]:
    """Parse 'float32[B, 16]' into ('float32', ['B', '16'])."""
    ann = ann.strip().strip("'\"")
    if "[" in ann:
        dtype = ann[:ann.index("[")]
        dims = ann[ann.index("[") + 1 : ann.index("]")]
        dim_list = [d.strip() for d in dims.split(",")]
        return dtype, dim_list
    return ann, []


def _aten_body_to_cpp(body_code: str, return_names: list[str]) -> list[str]:
    """Translate a uniquified Python aten body to C++ ATen ops.

    Handles the common patterns found in uniquified function bodies:
    - ``var: 'dtype[dims]' = aten.op(args)``
    - ``var: 'dtype[dims]' = aten.op.overload(args)``
    - ``var = aten.op(args)``
    - ``var: 'dtype[dims]' = operator.getitem(container, idx)``
    - ``return (var1, var2, ...)``

    Returns a list of C++ lines (without leading indent).
    """
    # Pattern: var (optional annotation) = aten.op(.overload)?(args)
    aten_re = re.compile(
        r"^\s*(\w+)"                       # var name
        r"(?:\s*:\s*'[^']*')?"             # optional annotation
        r"\s*=\s*"
        r"aten\.(\w+)(?:\.(\w+))?"         # aten.op or aten.op.overload
        r"\((.+)\)\s*$"                    # (args)
    )
    # Pattern: var (optional annotation) = operator.getitem(container, idx)
    getitem_re = re.compile(
        r"^\s*(\w+)"
        r"(?:\s*:\s*'[^']*')?"
        r"\s*=\s*"
        r"operator\.getitem\((\w+),\s*(\d+)\)\s*$"
    )
    # Pattern: return (var1, var2, ...)
    return_re = re.compile(r"^\s*return\s+\((.+?),?\)\s*$")

    # Arithmetic ops that look better as infix operators
    infix_ops = {
        "add": "+", "sub": "-", "mul": "*", "div": "/",
    }

    def _fix_args(args_str: str) -> str:
        """Translate Python literals in args to C++ equivalents."""
        s = args_str
        # [16] → {16}  (shape/size lists)
        s = re.sub(r'\[([^\[\]]*)\]', r'{\1}', s)
        # True/False → true/false
        s = re.sub(r'\bTrue\b', 'true', s)
        s = re.sub(r'\bFalse\b', 'false', s)
        # None → c10::nullopt
        s = re.sub(r'\bNone\b', 'c10::nullopt', s)
        return s

    cpp_lines: list[str] = []

    for line in body_code.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue

        # Match aten op
        m = aten_re.match(stripped)
        if m:
            var, op, overload, args = m.group(1), m.group(2), m.group(3), m.group(4)
            args = _fix_args(args)

            # Check for infix operators (aten.add.Tensor, aten.mul.Scalar, etc.)
            if op in infix_ops and overload in ("Tensor", "Scalar", None):
                parts = [a.strip() for a in args.split(",", 1)]
                if len(parts) == 2:
                    cpp_lines.append(f"auto {var} = {parts[0]} {infix_ops[op]} {parts[1]};")
                    continue

            cpp_lines.append(f"auto {var} = torch::{op}({args});")
            continue

        # Match operator.getitem
        m = getitem_re.match(stripped)
        if m:
            var, container, idx = m.group(1), m.group(2), m.group(3)
            cpp_lines.append(f"auto {var} = std::get<{idx}>({container});")
            continue

        # Match return
        m = return_re.match(stripped)
        if m:
            ret_vars = [v.strip() for v in m.group(1).split(",") if v.strip()]
            if len(ret_vars) == 1:
                cpp_lines.append(f"return {ret_vars[0]};")
            else:
                cpp_lines.append(f"return std::make_tuple({', '.join(ret_vars)});")
            continue

        # Fallback: emit as comment
        cpp_lines.append(f"// [unrecognized] {stripped}")

    return cpp_lines


def cuda_kernel_template(
    fn_name: str,
    params: list[dict],
    returns: list[dict],
    body_code: str = "",
) -> str:
    """Generate a CUDA C++ source template for a uniquified function.

    When *body_code* is provided (the Python aten ops from the uniquified
    function), it is auto-translated to equivalent C++ ATen ops, producing a
    **working** kernel that can be compiled and run immediately.  Users can
    then optimize the C++ code (fuse kernels, add shared memory, etc.)
    starting from a known-correct baseline.

    Args:
        fn_name: Name of the function (e.g. ``"residual_block"``).
        params: List of ``{"name": str, "annotation": str}`` dicts describing
            the function parameters.
        returns: List of ``{"name": str, "annotation": str}`` dicts describing
            the return values.
        body_code: The Python aten function body from the uniquified group.
            If provided, auto-translated to C++ ATen ops.

    Returns:
        A CUDA C++ source string ready to be passed to :func:`load_cuda`.
    """
    lines = []
    lines.append('#include <torch/extension.h>')
    lines.append('')

    # Build C++ function signature
    cpp_fn_name = f"fused_{fn_name}"

    # Determine return type
    if len(returns) == 0:
        ret_type = "void"
    elif len(returns) == 1:
        ret_type = "torch::Tensor"
    else:
        ret_type = f"std::tuple<{', '.join(['torch::Tensor'] * len(returns))}>"

    # Build parameter list with comments
    param_parts = []
    for i, p in enumerate(params):
        comma = "," if i < len(params) - 1 else ""
        comment = f"  // {p['annotation']}" if p.get("annotation") else ""
        param_parts.append(f"    torch::Tensor {p['name']}{comma}{comment}")

    sig = f"{ret_type} {cpp_fn_name}(\n"
    sig += "\n".join(param_parts)
    sig += "\n)"

    lines.append(sig + " {")

    # Body: auto-translate from aten Python ops if available
    return_names = [r["name"] for r in returns]
    if body_code.strip():
        cpp_body = _aten_body_to_cpp(body_code, return_names)
        if cpp_body:
            lines.append("    // Auto-generated from aten ops — edit to fuse/optimize.")
            lines.append("    // To write a raw CUDA kernel, use .data_ptr<T>() on tensors")
            lines.append("    // and launch with <<<blocks, threads>>>.")
            for cl in cpp_body:
                lines.append(f"    {cl}")
        else:
            lines.append("    // Could not auto-translate body. Fill in manually.")
            lines.append(f"    return torch::empty_like({params[0]['name']});")
    else:
        lines.append("    // TODO: implement fused kernel.")
        if len(returns) == 0:
            pass
        elif len(returns) == 1:
            lines.append(f"    return torch::empty_like({params[0]['name']});  // placeholder")
        else:
            out_names = ", ".join(
                f"torch::empty_like({params[0]['name']})" for _ in returns
            )
            lines.append(f"    return std::make_tuple({out_names});")

    lines.append("}")
    lines.append("")

    return "\n".join(lines)


def cuda_wrapper_template(
    fn_name: str,
    params: list[dict],
    returns: list[dict],
    cuda_var_name: str | None = None,
    module_var_name: str | None = None,
) -> str:
    """Generate the Python wrapper code that loads and calls the CUDA kernel.

    This produces the Python function body that replaces the original aten ops
    with a call to the compiled CUDA kernel.

    Args:
        fn_name: Function name (e.g. ``"residual_block"``).
        params: Parameter definitions from the uniquified function.
        returns: Return definitions from the uniquified function.
        cuda_var_name: Name of the CUDA source variable (default:
            ``f"_{fn_name.upper()}_CUDA"``).
        module_var_name: Name of the cached module variable (default:
            ``f"_{fn_name}_mod"``).

    Returns:
        Python source code for the wrapper function.
    """
    if cuda_var_name is None:
        cuda_var_name = f"_{fn_name.upper()}_CUDA"
    if module_var_name is None:
        module_var_name = f"_{fn_name}_mod"

    cpp_fn_name = f"fused_{fn_name}"
    param_names = [p["name"] for p in params]

    lines = []
    lines.append(f"{module_var_name} = None")
    lines.append("")
    lines.append(f"def {fn_name}(")
    for p in params:
        ann = f": '{p['annotation']}'" if p.get("annotation") else ""
        lines.append(f"    {p['name']}{ann},")

    # Return annotation
    if returns:
        ret_anns = []
        for r in returns:
            ret_anns.append(f"'{r['annotation']}'" if r.get("annotation") else "...")
        lines.append(f") -> tuple[{', '.join(ret_anns)}]:")
    else:
        lines.append("):")

    lines.append(f"    global {module_var_name}")
    lines.append(f"    if {module_var_name} is None:")
    lines.append(f'        {module_var_name} = load_cuda("{fn_name}", {cuda_var_name}, ["{cpp_fn_name}"])')

    # Build call
    call_args = ", ".join(param_names)
    if len(returns) <= 1:
        lines.append(f"    return ({module_var_name}.{cpp_fn_name}({call_args}),)")
    else:
        lines.append(f"    return {module_var_name}.{cpp_fn_name}({call_args})")

    return "\n".join(lines)
