"""Shared utilities for ref CUDA kernel tests."""
import os
import torch
from torch.utils.cpp_extension import load_inline

# Triton linker fix
if "LIBRARY_PATH" not in os.environ or "stubs" not in os.environ.get("LIBRARY_PATH", ""):
    import glob
    stubs = glob.glob("/usr/local/cuda*/targets/x86_64-linux/lib/stubs")
    if stubs:
        os.environ["LIBRARY_PATH"] = stubs[0] + ":" + os.environ.get("LIBRARY_PATH", "")

_cache = {}

def compile_cuda(name, cuda_src, functions):
    """Compile CUDA source via load_inline with caching.

    The cuda_src must contain C++ wrapper functions (not just __global__ kernels).
    load_inline needs forward declarations in cpp_sources for pybind to find them.
    """
    if name in _cache:
        return _cache[name]
    # Build forward declarations for pybind
    import re
    decls = []
    for fn in functions:
        # Find the function signature in cuda_src
        # Match: return_type func_name(args) {
        pattern = rf'([\w:<>\s]+)\s+{re.escape(fn)}\s*\(([^)]*)\)\s*\{{'
        m = re.search(pattern, cuda_src)
        if m:
            ret_type = m.group(1).strip()
            args = m.group(2).strip()
            decls.append(f"{ret_type} {fn}({args});")
    cpp_src = "\n".join(decls) if decls else ""
    ext = load_inline(name=name, cpp_sources=cpp_src, cuda_sources=[cuda_src],
                      functions=functions, verbose=False)
    _cache[name] = ext
    return ext


def check(name, result, expected, atol=1e-5, rtol=1e-5):
    """Compare result to expected, raise on mismatch."""
    if isinstance(result, (list, tuple)):
        for i, (r, e) in enumerate(zip(result, expected)):
            check(f"{name}[{i}]", r, e, atol, rtol)
        return
    if result.dtype != expected.dtype:
        result = result.to(expected.dtype)
    if expected.dtype in (torch.bool,):
        assert torch.equal(result, expected), (
            f"{name}: bool mismatch, got {result.sum()} true vs {expected.sum()} true")
        return
    if expected.dtype in (torch.long, torch.int32, torch.int64):
        assert torch.equal(result, expected), (
            f"{name}: int mismatch")
        return
    diff = (result.float() - expected.float()).abs().max().item()
    assert torch.allclose(result.float(), expected.float(), atol=atol, rtol=rtol), (
        f"{name}: max diff {diff:.2e} (atol={atol})")
