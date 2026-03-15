"""Track and reproduce custom op registrations in exported aten files.

Three hooks intercept ALL ways custom ops can be registered in PyTorch:
  1. torch.library.Library()              — Python-side registration
  2. torch.ops.load_library()             — explicit C++ extension loading
  3. ExtensionFileLoader.create_module()  — C++ TORCH_LIBRARY via Python import

Each hook records the namespace, the calling module/function/args, and .so path
so that exported aten files can reproduce the registration automatically.

The tracker is installed at import time (before any user code registers ops).
"""

from __future__ import annotations

import logging
import os
import re
import sys
from dataclasses import dataclass
from io import StringIO

import torch

logger = logging.getLogger(__name__)

# Built-in torch.ops namespaces that don't need external imports
BUILTIN_OP_NAMESPACES = frozenset({
    "aten", "prim", "prims", "quantized", "profiler", "_quantized",
    "_c10d_functional", "c10d_functional", "_inductor_test",
    "debug_mode_ops", "debugprims", "export", "flex_lib", "fsdp",
    "higher_order", "inductor", "mkl", "mkldnn", "onednn", "onnx",
    "onnx_symbolic", "rngprims", "streams", "triton",
})


# ── Data classes ──────────────────────────────────────────────────────

@dataclass
class OpRegistration:
    """Record of how a custom op namespace was registered."""
    namespace: str
    module_name: str | None = None    # __name__ of the registering module
    module_file: str | None = None    # __file__ of the registering module
    func_name: str | None = None      # function that registered (None = module level)
    func_qualname: str | None = None  # "module.func" for code gen
    func_args: str | None = None      # repr of positional args (from f_locals)
    so_path: str | None = None        # .so path (for load_library / extension import)


@dataclass
class OpProvider:
    """Resolved info for how to load a custom op namespace in generated code."""
    namespace: str
    example_op: str                   # one op name for runtime verification
    module_name: str | None = None
    module_file: str | None = None
    func_name: str | None = None      # None = module-level (import alone works)
    func_qualname: str | None = None  # "module.func" for the generated call
    func_args: str | None = None      # repr of arguments for the init call
    so_path: str | None = None
    so_filename: str | None = None
    hf_repo_dir: str | None = None
    hf_repo_id: str | None = None


# ── Global registry ──────────────────────────────────────────────────

op_registrations: dict[str, OpRegistration] = {}


# ── Tracker hooks ────────────────────────────────────────────────────

def _detect_new_namespaces(before: set[str], after: set[str]) -> set[str]:
    """Find custom (non-builtin) namespaces that appeared between two op snapshots."""
    new_nss: set[str] = set()
    for op_name in after - before:
        if "::" in op_name:
            ns = op_name.split("::")[0]
            if ns not in BUILTIN_OP_NAMESPACES:
                new_nss.add(ns)
    return new_nss


def _snapshot_ops() -> set[str]:
    try:
        return set(torch._C._dispatch_get_all_op_names())
    except Exception:
        return set()


def _install_op_tracker():
    """Install hooks on torch.library.Library, torch.ops.load_library, and
    importlib.machinery.ExtensionFileLoader to track all op registrations."""

    # ── Hook 1: torch.library.Library.__init__ ────────────────────
    # Catches Python-side registration (e.g. cuda-side-boost).
    _orig_library_init = torch.library.Library.__init__

    def _tracking_library_init(self, ns, kind, *args, **kwargs):
        _orig_library_init(self, ns, kind, *args, **kwargs)
        if ns in BUILTIN_OP_NAMESPACES or ns in op_registrations:
            return

        # Walk up the call stack to find the user's module and function
        frame = sys._getframe(1)
        module_name = frame.f_globals.get("__name__")
        module_file = frame.f_globals.get("__file__")
        func_name = None
        func_args = None
        f = frame
        while f is not None:
            code_name = f.f_code.co_name
            mod = f.f_globals.get("__name__", "")
            if code_name != "<module>" and not mod.startswith(("torch.", "torch_graph.")):
                func_name = code_name
                module_name = f.f_globals.get("__name__")
                module_file = f.f_globals.get("__file__")
                # Capture function arguments from f_locals using the
                # code object's parameter names (preserves call order)
                try:
                    code = f.f_code
                    param_names = code.co_varnames[:code.co_argcount]
                    locals_copy = f.f_locals
                    args_list = []
                    for pname in param_names:
                        if pname == "self":
                            continue
                        val = locals_copy.get(pname)
                        args_list.append(repr(val))
                    func_args = ", ".join(args_list) if args_list else None
                except Exception:
                    pass
                break
            if code_name == "<module>" and not mod.startswith(("torch.", "torch_graph.")):
                break
            f = f.f_back

        op_registrations[ns] = OpRegistration(
            namespace=ns, module_name=module_name, module_file=module_file,
            func_name=func_name,
            func_qualname=f"{module_name}.{func_name}" if func_name and module_name else None,
            func_args=func_args,
        )
        logger.debug("Tracked Library(%r) from %s%s(%s)",
                      ns, module_name,
                      f".{func_name}" if func_name else "",
                      func_args or "")

    torch.library.Library.__init__ = _tracking_library_init

    # ── Hook 2: torch.ops.load_library ────────────────────────────
    # Catches explicit torch.ops.load_library("/path/to/ops.so").
    _orig_load_library = torch.ops.load_library

    def _tracking_load_library(path):
        before = _snapshot_ops()
        _orig_load_library(path)
        after = _snapshot_ops()

        frame = sys._getframe(1)
        caller_module = frame.f_globals.get("__name__")
        caller_file = frame.f_globals.get("__file__")
        for ns in _detect_new_namespaces(before, after):
            if ns not in op_registrations:
                op_registrations[ns] = OpRegistration(
                    namespace=ns, module_name=caller_module,
                    module_file=caller_file, so_path=path,
                )
                logger.debug("Tracked load_library(%r) -> torch.ops.%s.*", path, ns)

    torch.ops.load_library = _tracking_load_library

    # ── Hook 3: ExtensionFileLoader.create_module ─────────────────
    # Catches C++ extensions loaded via normal Python import that register
    # ops via TORCH_LIBRARY in their init function (e.g. HF kernels).
    from importlib.machinery import ExtensionFileLoader
    _orig_create_module = ExtensionFileLoader.create_module

    def _tracking_create_module(self, spec):
        before = _snapshot_ops()
        result = _orig_create_module(self, spec)
        after = _snapshot_ops()

        so_path = getattr(spec, "origin", None)
        for ns in _detect_new_namespaces(before, after):
            if ns not in op_registrations:
                op_registrations[ns] = OpRegistration(
                    namespace=ns,
                    module_name=spec.name if spec else None,
                    module_file=so_path,
                    so_path=so_path,
                )
                logger.debug("Tracked extension import %s -> torch.ops.%s.*",
                              spec.name if spec else "?", ns)
        return result

    ExtensionFileLoader.create_module = _tracking_create_module


# Install at import time.
_install_op_tracker()


# ── Graph scanning ───────────────────────────────────────────────────

def find_custom_op_namespaces(capture) -> dict[str, str]:
    """Scan capture graphs for non-builtin torch.ops namespaces.

    Returns a dict mapping namespace → one example op name from that namespace
    (e.g. {"flash_attn_3": "_flash_attn_forward"}).  The example op is used
    for runtime verification since torch.ops creates namespaces lazily.
    """
    ns_to_op: dict[str, str] = {}
    all_graphs = list(capture.forward_graphs) + list(capture.backward_graphs)
    for ag in all_graphs:
        for node in ag.graph_module.graph.nodes:
            if node.op == "call_function" and hasattr(node.target, "name") and callable(getattr(node.target, "name", None)):
                try:
                    name = node.target.name()
                    if "::" in name:
                        ns, op_name = name.split("::", 1)
                        if ns not in BUILTIN_OP_NAMESPACES and ns not in ns_to_op:
                            ns_to_op[ns] = op_name
                except Exception:
                    pass
    return ns_to_op


# ── Provider resolution ──────────────────────────────────────────────

def build_op_providers(ns_to_op: dict[str, str]) -> list[OpProvider]:
    """For each custom namespace, resolve how to load it in generated code.

    Relies entirely on op_registrations populated by the tracker hooks.
    """
    if not ns_to_op:
        return []

    result: list[OpProvider] = []
    for ns in sorted(ns_to_op):
        prov = OpProvider(namespace=ns, example_op=ns_to_op[ns])
        reg = op_registrations.get(ns)
        if reg:
            prov.module_name = reg.module_name
            prov.module_file = reg.module_file
            prov.func_name = reg.func_name
            prov.func_qualname = reg.func_qualname
            prov.func_args = reg.func_args
            prov.so_path = reg.so_path
            if prov.so_path:
                prov.so_filename = os.path.basename(prov.so_path)
            # Enrich with HF metadata if .so is from HF cache
            path_to_check = prov.so_path or prov.module_file or ""
            hf_match = re.search(r"(models--.+?)/snapshots/", path_to_check)
            if hf_match:
                prov.hf_repo_dir = hf_match.group(1)
                parts = prov.hf_repo_dir.split("--", 1)
                if len(parts) == 2:
                    prov.hf_repo_id = parts[1].replace("--", "/")
        result.append(prov)

    return result


# ── Code emission ────────────────────────────────────────────────────

def emit_custom_op_imports(buf: StringIO, providers: list[OpProvider]) -> None:
    """Emit import/load code for custom op namespaces in generated aten files."""
    if not providers:
        return

    has_so = any(p.so_path for p in providers)

    buf.write("# ── Custom operator libraries ──────────────────────────────────────\n")
    buf.write("# These ops were registered by external libraries at capture time.\n")

    # Emit .so loader with Python package import fallback.
    # load_library works for standalone C++ extensions (TORCH_LIBRARY static init).
    # Some .so files are Python extension modules inside packages where ops are
    # registered by Python code (@torch.library.custom_op) — for those,
    # load_library succeeds but doesn't register the ops.  The fallback detects
    # this and imports the containing Python package instead.
    if has_so:
        buf.write("def _load_custom_ops(so_filename, captured_path, ns, example_op):\n")
        buf.write('    """Load custom ops from a .so, falling back to Python package import."""\n')
        buf.write("    import glob as _glob, importlib, sys\n")
        buf.write("    so_path = captured_path if os.path.exists(captured_path) else None\n")
        buf.write("    if so_path is None:\n")
        buf.write('        hf_cache = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")\n')
        buf.write('        matches = _glob.glob(os.path.join(hf_cache, "**", so_filename), recursive=True)\n')
        buf.write("        if matches:\n")
        buf.write("            so_path = matches[0]\n")
        buf.write("    if so_path is None:\n")
        buf.write('        raise FileNotFoundError(f"Cannot find {so_filename} (captured at {captured_path})")\n')
        buf.write("    torch.ops.load_library(so_path)\n")
        buf.write("    if hasattr(getattr(torch.ops, ns, None), example_op):\n")
        buf.write("        return\n")
        buf.write("    # load_library didn't register the ops — try Python package import\n")
        buf.write("    so_dir = os.path.dirname(so_path)\n")
        buf.write('    if os.path.isfile(os.path.join(so_dir, "__init__.py")):\n')
        buf.write("        pkg_name = os.path.basename(so_dir)\n")
        buf.write("        parent_dir = os.path.dirname(so_dir)\n")
        buf.write("        if parent_dir not in sys.path:\n")
        buf.write("            sys.path.insert(0, parent_dir)\n")
        buf.write("        importlib.import_module(pkg_name)\n")
        buf.write("\n")

    for prov in providers:
        if prov.so_path:
            buf.write(f"_load_custom_ops({prov.so_filename!r}, {prov.so_path!r}, {prov.namespace!r}, {prov.example_op!r})")
            buf.write(f"  # torch.ops.{prov.namespace}.*\n")
        elif prov.func_qualname:
            # Python module + init function needed
            top_module = prov.module_name.split(".")[0]
            args_str = prov.func_args or ""
            buf.write(f"import {top_module}; {prov.func_qualname}({args_str})")
            buf.write(f"  # registers torch.ops.{prov.namespace}.*\n")
        elif prov.module_name:
            # Python module — import alone registers the ops
            top_module = prov.module_name.split(".")[0]
            buf.write(f"import {top_module}")
            buf.write(f"  # registers torch.ops.{prov.namespace}.*\n")
        else:
            buf.write(f"# TODO: load the library that provides torch.ops.{prov.namespace}.*\n")

    # Runtime check using a specific op from each namespace
    buf.write("\n# Verify custom op namespaces are actually registered\n")
    buf.write("_required_ops = %r\n" % {p.namespace: p.example_op for p in providers})
    buf.write("for _ns, _op in _required_ops.items():\n")
    buf.write("    if not hasattr(getattr(torch.ops, _ns), _op):\n")
    buf.write('        raise RuntimeError(\n')
    buf.write('            f"torch.ops.{_ns}.{_op} is not registered. "\n')
    buf.write('            f"Import or load the library that provides torch.ops.{_ns}.* before running this script."\n')
    buf.write("        )\n")
    buf.write("\n")
