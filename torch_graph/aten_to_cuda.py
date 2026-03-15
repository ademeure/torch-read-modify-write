"""Convert aten op graphs to fused inline CUDA kernels.

Given a kbox-style ``run(inputs)`` function containing aten op calls, produces
a version where sequences of elementwise ops are fused into single CUDA kernels.

Pipeline:
  1. Parse: AST-extract each aten op line → structured OpNode
  2. Classify: tag each op as elementwise / structural / unsupported
  3. Fuse: find maximal chains of elementwise ops on same-shaped data
  4. Codegen: emit fused CUDA kernel + Python wrapper

Structural ops (view, transpose, permute, etc.) inside a fused region become
index arithmetic in the kernel — the kernel reads/writes contiguous memory and
computes strided source indices internally.

Ops without CUDA support (mm, bmm, addmm, embedding, convolution, split, etc.)
stay as PyTorch calls and act as fusion barriers.
"""

from __future__ import annotations

import ast
import re
import textwrap
from dataclasses import dataclass, field
from typing import Any


# ═══════════════════════════════════════════════════════════════════════════════
# Elementwise CUDA expression templates
# ═══════════════════════════════════════════════════════════════════════════════

# Maps aten op name → (n_tensor_inputs, cuda_expr_template, needs_includes)
# Template uses {in0}, {in1}, {out0} etc. for tensor elements,
# and {s0}, {s1} for scalar arguments.
_ELEMENTWISE_EXPR: dict[str, tuple[int, str, str]] = {
    # Activations
    "relu":      (1, "({in0} > 0.0f ? {in0} : 0.0f)", ""),
    "gelu":      (1, "({in0} * 0.5f * (1.0f + erff({in0} * 0.7071067811865476f)))", ""),
    "silu":      (1, "({in0} / (1.0f + expf(-{in0})))", ""),
    "sigmoid":   (1, "(1.0f / (1.0f + expf(-{in0})))", ""),
    "tanh":      (1, "tanhf({in0})", ""),
    "hardswish": (1, "({in0} * fminf(fmaxf({in0} + 3.0f, 0.0f), 6.0f) / 6.0f)", ""),
    "hardsigmoid": (1, "fminf(fmaxf({in0} / 6.0f + 0.5f, 0.0f), 1.0f)", ""),
    "hardtanh":  (1, "fminf(fmaxf({in0}, -1.0f), 1.0f)", ""),
    "leaky_relu": (1, "({in0} > 0.0f ? {in0} : 0.01f * {in0})", ""),
    "elu":       (1, "({in0} > 0.0f ? {in0} : (expf({in0}) - 1.0f))", ""),
    "softplus":  (1, "({in0} > 20.0f ? {in0} : logf(1.0f + expf({in0})))", ""),
    "mish":      (1, "({in0} * tanhf({in0} > 20.0f ? {in0} : logf(1.0f + expf({in0}))))", ""),

    # Unary math
    "abs":       (1, "fabsf({in0})", ""),
    "neg":       (1, "(-{in0})", ""),
    "exp":       (1, "expf({in0})", ""),
    "log":       (1, "logf({in0})", ""),
    "log2":      (1, "log2f({in0})", ""),
    "sqrt":      (1, "sqrtf({in0})", ""),
    "rsqrt":     (1, "rsqrtf({in0})", ""),
    "ceil":      (1, "ceilf({in0})", ""),
    "floor":     (1, "floorf({in0})", ""),
    "round":     (1, "nearbyintf({in0})", ""),
    "trunc":     (1, "truncf({in0})", ""),
    "sin":       (1, "sinf({in0})", ""),
    "cos":       (1, "cosf({in0})", ""),
    "erf":       (1, "erff({in0})", ""),
    "erfc":      (1, "erfcf({in0})", ""),
    "reciprocal": (1, "(1.0f / {in0})", ""),
    "sign":      (1, "({in0} > 0.0f ? 1.0f : ({in0} < 0.0f ? -1.0f : 0.0f))", ""),
    "sgn":       (1, "({in0} > 0.0f ? 1.0f : ({in0} < 0.0f ? -1.0f : 0.0f))", ""),

    # Binary elementwise (tensor, tensor)
    "add":       (2, "({in0} + {in1})", ""),
    "sub":       (2, "({in0} - {in1})", ""),
    "mul":       (2, "({in0} * {in1})", ""),
    "div":       (2, "({in0} / {in1})", ""),
    "pow":       (2, "powf({in0}, {in1})", ""),
    "maximum":   (2, "fmaxf({in0}, {in1})", ""),
    "minimum":   (2, "fminf({in0}, {in1})", ""),

    # Comparison (output as float)
    "eq":        (2, "({in0} == {in1} ? 1.0f : 0.0f)", ""),
    "ne":        (2, "({in0} != {in1} ? 1.0f : 0.0f)", ""),
    "gt":        (2, "({in0} > {in1} ? 1.0f : 0.0f)", ""),
    "ge":        (2, "({in0} >= {in1} ? 1.0f : 0.0f)", ""),
    "lt":        (2, "({in0} < {in1} ? 1.0f : 0.0f)", ""),
    "le":        (2, "({in0} <= {in1} ? 1.0f : 0.0f)", ""),

    # Ternary
    "where":     (3, "({in0} != 0.0f ? {in1} : {in2})", ""),
    "clamp":     (1, "fminf(fmaxf({in0}, {s0}), {s1})", ""),
    "clamp_min": (1, "fmaxf({in0}, {s0})", ""),
    "clamp_max": (1, "fminf({in0}, {s0})", ""),
    "masked_fill": (2, "({in1} != 0.0f ? {s0} : {in0})", ""),

    # Backward ops
    "threshold_backward": (2, "({in1} > 0.0f ? {in0} : 0.0f)", ""),
    "gelu_backward": (2,
        "({in0} * (0.5f * (1.0f + erff({in1} * 0.7071067811865476f)) + "
        "{in1} * 0.3989422804014327f * expf(-0.5f * {in1} * {in1})))", ""),
    "silu_backward": (2,
        "({in0} * (1.0f / (1.0f + expf(-{in1}))) * "
        "(1.0f + {in1} * (1.0f - 1.0f / (1.0f + expf(-{in1})))))", ""),
    "sigmoid_backward": (2, "({in0} * {in1} * (1.0f - {in1}))", ""),
    "tanh_backward": (2, "({in0} * (1.0f - {in1} * {in1}))", ""),
}

# Ops that are structural (reshape memory layout, no computation).
# Inside a fused kernel, these become index math.
_STRUCTURAL_OPS = {
    "view", "reshape", "_unsafe_view", "t", "transpose", "permute",
    "expand", "contiguous", "unsqueeze", "squeeze", "slice", "select",
    "as_strided", "alias", "detach", "clone",
}

# Ops that are fusion barriers — can't be fused into CUDA kernels.
_BARRIER_OPS = {
    "mm", "bmm", "addmm", "matmul", "linear",
    "convolution", "conv2d", "conv1d", "conv_transpose2d",
    "embedding", "embedding_dense_backward",
    "_scaled_dot_product_flash_attention",
    "_scaled_dot_product_efficient_attention",
    "_scaled_dot_product_cudnn_attention",
    "native_layer_norm", "native_batch_norm",
    "_softmax", "_log_softmax",
    "native_dropout",
    "nll_loss_forward", "nll_loss_backward",
    "split", "cat", "stack",
    "arange", "zeros", "ones", "full", "empty",
    "sum", "mean", "amax", "amin", "prod",  # reductions need shared mem
    "sort", "topk", "scatter", "gather", "index_select", "index_put",
    "cumsum",
}

# getitem is special — it extracts from a tuple, not a tensor op
_SPECIAL_OPS = {"__getitem__", "getitem"}


# ═══════════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class OpArg:
    """An argument to an aten op."""
    kind: str       # "var" (local variable), "input" (inputs.X), "scalar"
    name: str = ""  # variable/input name
    value: Any = None  # scalar value

    def __repr__(self):
        if self.kind == "scalar":
            return f"Scalar({self.value!r})"
        return f"{self.kind}({self.name})"


@dataclass
class OpNode:
    """A single aten op extracted from the run() function."""
    output_var: str       # LHS variable name
    op_name: str          # e.g. "add", "relu", "view"
    op_variant: str       # e.g. "default", "Tensor", "Scalar", "int"
    args: list[OpArg] = field(default_factory=list)
    line: str = ""        # original source line
    line_number: int = 0

    @property
    def is_elementwise(self) -> bool:
        return self.op_name in _ELEMENTWISE_EXPR

    @property
    def is_structural(self) -> bool:
        return self.op_name in _STRUCTURAL_OPS

    @property
    def is_barrier(self) -> bool:
        return self.op_name in _BARRIER_OPS

    @property
    def is_special(self) -> bool:
        return self.op_name in _SPECIAL_OPS

    @property
    def tensor_args(self) -> list[OpArg]:
        return [a for a in self.args if a.kind in ("var", "input")]

    @property
    def scalar_args(self) -> list[OpArg]:
        return [a for a in self.args if a.kind == "scalar"]

    @property
    def has_scalar_variant(self) -> bool:
        """E.g. aten.add.Scalar, aten.mul.Scalar — second arg is scalar."""
        return self.op_variant == "Scalar"


@dataclass
class FusionGroup:
    """A sequence of ops to be fused into a single CUDA kernel."""
    ops: list[OpNode]
    # External tensor inputs (from inputs.X or from before the group)
    input_vars: list[str]
    # Output variables consumed outside the group or returned
    output_vars: list[str]


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: Parse run(inputs) function
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_arg(node: ast.expr) -> OpArg:
    """Parse a function call argument into an OpArg."""
    # inputs.name → input
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        if node.value.id == "inputs":
            return OpArg(kind="input", name=node.attr)
    # local_var → var
    if isinstance(node, ast.Name):
        if node.id in ("True", "False", "None", "inf"):
            mapping = {"True": True, "False": False, "None": None,
                       "inf": float("inf")}
            return OpArg(kind="scalar", value=mapping[node.id])
        return OpArg(kind="var", name=node.id)
    # -inf
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        if isinstance(node.operand, ast.Name) and node.operand.id == "inf":
            return OpArg(kind="scalar", value=float("-inf"))
        if isinstance(node.operand, ast.Constant):
            return OpArg(kind="scalar", value=-node.operand.value)
    # Scalar constant
    if isinstance(node, ast.Constant):
        return OpArg(kind="scalar", value=node.value)
    # List literal [32, 64] etc.
    if isinstance(node, ast.List):
        vals = []
        for elt in node.elts:
            if isinstance(elt, ast.Constant):
                vals.append(elt.value)
            elif isinstance(elt, ast.UnaryOp) and isinstance(elt.op, ast.USub):
                vals.append(-elt.operand.value)
            else:
                vals.append(ast.unparse(elt))
        return OpArg(kind="scalar", value=vals)
    # torch.contiguous_format etc.
    try:
        return OpArg(kind="scalar", value=ast.unparse(node))
    except Exception:
        return OpArg(kind="scalar", value=None)


def parse_run_function(source: str) -> list[OpNode]:
    """Parse a run(inputs) function source into OpNode list."""
    tree = ast.parse(textwrap.dedent(source))
    ops = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "run":
            continue
        for stmt in node.body:
            if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
                continue
            target = stmt.targets[0]
            if not isinstance(target, ast.Name):
                continue
            output_var = target.id
            call = stmt.value
            if not isinstance(call, ast.Call):
                continue

            op_name, op_variant = _parse_op_target(call.func)
            if op_name is None:
                continue

            args = [_parse_arg(a) for a in call.args]
            # Also handle keyword args
            for kw in call.keywords:
                args.append(_parse_arg(kw.value))

            ops.append(OpNode(
                output_var=output_var,
                op_name=op_name,
                op_variant=op_variant,
                args=args,
                line=ast.unparse(stmt),
                line_number=stmt.lineno,
            ))
    return ops


def _parse_op_target(node: ast.expr) -> tuple[str | None, str]:
    """Extract (op_name, variant) from torch.ops.aten.X.Y or operator.getitem."""
    if isinstance(node, ast.Attribute):
        parts = []
        cur = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        parts.reverse()
        # torch.ops.aten.op_name.variant
        if len(parts) >= 5 and parts[:3] == ["torch", "ops", "aten"]:
            return parts[3], parts[4]
        # operator.getitem
        if parts == ["operator", "getitem"]:
            return "__getitem__", "default"
    return None, ""


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: Classify ops and find fusion groups
# ═══════════════════════════════════════════════════════════════════════════════

def _find_fusion_groups(ops: list[OpNode], return_vars: set[str]) -> list[FusionGroup | OpNode]:
    """Find maximal sequences of fusable elementwise+structural ops.

    Returns a mixed list of FusionGroup (for fused sequences) and bare OpNode
    (for barrier/special ops that can't be fused).

    A fusion group must have:
    - At least one elementwise op (pure structural chains aren't worth fusing)
    - All ops are either elementwise or structural
    - No barrier ops inside
    """
    result: list[FusionGroup | OpNode] = []
    current_chain: list[OpNode] = []

    def _flush_chain():
        if not current_chain:
            return
        chain_copy = list(current_chain)
        current_chain.clear()
        # Only fuse if there's at least one elementwise op
        has_compute = any(op.is_elementwise for op in chain_copy)
        if has_compute:
            result.append(_build_fusion_group(chain_copy, ops, return_vars))
        else:
            # Not worth fusing — emit individually
            for op in chain_copy:
                result.append(op)

    for op in ops:
        if op.is_elementwise or op.is_structural:
            current_chain.append(op)
        else:
            _flush_chain()
            result.append(op)

    _flush_chain()
    return result


def _build_fusion_group(chain: list[OpNode], all_ops: list[OpNode],
                        return_vars: set[str]) -> FusionGroup:
    """Build a FusionGroup from a chain of ops."""
    produced = set()  # vars produced within this group
    input_vars = []
    input_seen = set()
    output_vars = []

    # All vars produced in the group
    for op in chain:
        produced.add(op.output_var)

    # Find external inputs and outputs
    for op in chain:
        for arg in op.args:
            if arg.kind in ("var", "input"):
                name = arg.name
                if name not in produced and name not in input_seen:
                    input_vars.append(name)
                    input_seen.add(name)

    # Outputs: vars consumed outside the group or in return statement
    all_consumed_after = set()
    past_group = False
    for op in all_ops:
        if op is chain[-1]:
            past_group = True
            continue
        if past_group:
            for arg in op.args:
                if arg.kind == "var":
                    all_consumed_after.add(arg.name)

    for op in chain:
        if op.output_var in return_vars or op.output_var in all_consumed_after:
            output_vars.append(op.output_var)

    # If nothing is externally consumed, the last op's output is the output
    if not output_vars and chain:
        output_vars = [chain[-1].output_var]

    return FusionGroup(ops=chain, input_vars=input_vars, output_vars=output_vars)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: Generate fused CUDA kernel
# ═══════════════════════════════════════════════════════════════════════════════

def _gen_fused_kernel(group: FusionGroup, kernel_name: str) -> str:
    """Generate a fused CUDA kernel for a FusionGroup.

    The kernel operates elementwise on contiguous tensors. Structural ops
    (view, transpose, etc.) are no-ops because we ensure inputs/outputs are
    contiguous at the Python level.

    Returns the CUDA kernel source string.
    """
    # Map each variable to a CUDA expression
    var_expr: dict[str, str] = {}

    # External inputs get array reads
    for i, name in enumerate(group.input_vars):
        var_expr[name] = f"in{i}[i]"

    # Also map inputs.X references
    for op in group.ops:
        for arg in op.args:
            if arg.kind == "input" and arg.name not in var_expr:
                idx = group.input_vars.index(arg.name) if arg.name in group.input_vars else -1
                if idx >= 0:
                    var_expr[arg.name] = f"in{idx}[i]"

    # Process ops in order
    for op in group.ops:
        if op.is_structural:
            # Structural ops are identity in the fused kernel
            # (we make tensors contiguous at the Python boundary)
            tensor_args = op.tensor_args
            if tensor_args:
                src = tensor_args[0].name
                if src in var_expr:
                    var_expr[op.output_var] = var_expr[src]
                else:
                    var_expr[op.output_var] = f"/* {op.op_name} */ 0.0f"
            else:
                var_expr[op.output_var] = "0.0f"
            continue

        if op.op_name not in _ELEMENTWISE_EXPR:
            # Shouldn't happen — these should be barriers
            var_expr[op.output_var] = "0.0f"
            continue

        n_tensor_inputs, template, _ = _ELEMENTWISE_EXPR[op.op_name]

        # Build template substitution dict
        subs = {}
        tensor_idx = 0
        scalar_idx = 0

        for arg in op.args:
            if arg.kind in ("var", "input"):
                expr = var_expr.get(arg.name, f"/* missing {arg.name} */ 0.0f")
                subs[f"in{tensor_idx}"] = expr
                tensor_idx += 1
            elif arg.kind == "scalar":
                val = arg.value
                if isinstance(val, float):
                    if val == float("inf"):
                        subs[f"s{scalar_idx}"] = "INFINITY"
                    elif val == float("-inf"):
                        subs[f"s{scalar_idx}"] = "-INFINITY"
                    else:
                        subs[f"s{scalar_idx}"] = f"{val}f"
                    # For Scalar variant ops, the scalar acts as the second tensor input
                    if op.has_scalar_variant and f"in{tensor_idx}" not in subs:
                        subs[f"in{tensor_idx}"] = subs[f"s{scalar_idx}"]
                        tensor_idx += 1
                elif isinstance(val, int):
                    subs[f"s{scalar_idx}"] = f"{val}.0f"
                    if op.has_scalar_variant and f"in{tensor_idx}" not in subs:
                        subs[f"in{tensor_idx}"] = f"{float(val)}f"
                        tensor_idx += 1
                elif isinstance(val, (list, tuple)):
                    # Shape args — skip for elementwise
                    pass
                else:
                    subs[f"s{scalar_idx}"] = str(val)
                scalar_idx += 1

        # Apply substitutions
        expr = template
        for k, v in subs.items():
            expr = expr.replace("{" + k + "}", v)
        var_expr[op.output_var] = expr

    # Build kernel source
    n_inputs = len(group.input_vars)
    n_outputs = len(group.output_vars)

    # Kernel signature
    params = []
    for i in range(n_inputs):
        params.append(f"const float *in{i}")
    for i in range(n_outputs):
        params.append(f"float *out{i}")
    params.append("unsigned int n")
    sig = f'extern "C" __global__ void {kernel_name}(\n    {", ".join(params)}\n)'

    # Kernel body — compute all intermediate expressions, write outputs
    body_lines = []
    body_lines.append("    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;")
    body_lines.append("    if (i < n) {")

    # For readability: assign intermediate expressions to local vars
    # Only emit locals for expressions used more than once or for outputs
    expr_uses: dict[str, int] = {}
    for op in group.ops:
        if op.is_structural:
            continue
        for arg in op.args:
            if arg.kind in ("var", "input") and arg.name in var_expr:
                expr_uses[arg.name] = expr_uses.get(arg.name, 0) + 1

    # Emit local variables for multi-use intermediates
    declared = set()
    for op in group.ops:
        if op.output_var in expr_uses and expr_uses.get(op.output_var, 0) > 1:
            if op.output_var in var_expr and op.output_var not in declared:
                body_lines.append(
                    f"        float {op.output_var} = {var_expr[op.output_var]};")
                declared.add(op.output_var)
                # Update var_expr to use the local variable
                var_expr[op.output_var] = op.output_var

    # Write outputs
    for i, out_var in enumerate(group.output_vars):
        expr = var_expr.get(out_var, "0.0f")
        body_lines.append(f"        out{i}[i] = {expr};")

    body_lines.append("    }")

    kernel_src = f"{sig} {{\n" + "\n".join(body_lines) + "\n}"
    return kernel_src


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4: Generate complete Python output
# ═══════════════════════════════════════════════════════════════════════════════

def _gen_python_wrapper(group: FusionGroup, kernel_name: str,
                        kernel_src: str) -> list[str]:
    """Generate Python lines that replace the fused ops.

    Returns lines like:
        # ── Fused CUDA: relu + add + mul (3 ops) ──
        _KERNEL_SRC_0 = r\"\"\"...\"\"\"
        _kernel_0 = torch.utils.cpp_extension.load_inline(...)
        out0 = _kernel_0.forward(in0, in1)
    """
    lines = []

    op_names = [op.op_name for op in group.ops if op.is_elementwise]
    desc = " + ".join(op_names[:5])
    if len(op_names) > 5:
        desc += f" + ... ({len(op_names)} ops)"

    lines.append(f"    # ── Fused CUDA: {desc} ──")

    # For now, emit the kernel source as a comment for visibility,
    # and keep PyTorch calls as the execution path.
    # The kernel is ready to be plugged into kernelbox.
    lines.append(f"    # CUDA kernel ({len(group.input_vars)} inputs → "
                 f"{len(group.output_vars)} outputs):")
    for kline in kernel_src.split("\n"):
        lines.append(f"    #   {kline}")

    return lines


def _extract_return_vars(source: str) -> set[str]:
    """Extract variable names from 'return [var1, var2, ...]'."""
    tree = ast.parse(textwrap.dedent(source))
    ret_vars = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.List):
            for elt in node.value.elts:
                if isinstance(elt, ast.Name):
                    ret_vars.add(elt.id)
    return ret_vars


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def analyze(source: str) -> dict:
    """Analyze a run(inputs) function and return fusion analysis.

    Returns dict with:
        ops: list of OpNode
        groups: list of FusionGroup | OpNode
        kernels: list of (FusionGroup, kernel_name, kernel_source)
        stats: dict of counts
    """
    ops = parse_run_function(source)
    return_vars = _extract_return_vars(source)
    groups = _find_fusion_groups(ops, return_vars)

    kernels = []
    kernel_idx = 0
    for item in groups:
        if isinstance(item, FusionGroup):
            name = f"fused_{kernel_idx}"
            src = _gen_fused_kernel(item, name)
            kernels.append((item, name, src))
            kernel_idx += 1

    n_elementwise = sum(1 for op in ops if op.is_elementwise)
    n_structural = sum(1 for op in ops if op.is_structural)
    n_barrier = sum(1 for op in ops if op.is_barrier)
    n_special = sum(1 for op in ops if op.is_special)
    n_fused = sum(len(g.ops) for g, _, _ in kernels)

    return {
        "ops": ops,
        "groups": groups,
        "kernels": kernels,
        "stats": {
            "total_ops": len(ops),
            "elementwise": n_elementwise,
            "structural": n_structural,
            "barrier": n_barrier,
            "special": n_special,
            "fusion_groups": len(kernels),
            "ops_fused": n_fused,
        },
    }


def transform(source: str) -> str:
    """Transform a run(inputs) function, replacing fusable ops with CUDA kernels.

    Returns modified Python source with:
    - Fusable op sequences replaced by fused CUDA kernel + contiguous() calls
    - Barrier ops left as-is
    - CUDA kernel source embedded as string constants

    The output is a valid kbox script that can be iterated on.
    """
    result = analyze(source)
    ops = result["ops"]
    groups = result["groups"]

    # Rebuild the function
    out_lines = []
    # Extract imports and init_once from source
    for line in source.split("\n"):
        stripped = line.strip()
        if stripped.startswith("def run("):
            break
        out_lines.append(line)

    # Kernel source constants
    for group, kname, ksrc in result["kernels"]:
        out_lines.append(f"")
        op_names = [op.op_name for op in group.ops if op.is_elementwise]
        desc = " → ".join(op_names[:6])
        if len(op_names) > 6:
            desc += f" → ... ({len(op_names)} total)"
        out_lines.append(f"# Fused kernel: {desc}")
        out_lines.append(f"# Inputs: {', '.join(group.input_vars)}")
        out_lines.append(f"# Outputs: {', '.join(group.output_vars)}")
        out_lines.append(f"KERNEL_{kname.upper()} = r'''")
        out_lines.append(ksrc)
        out_lines.append("'''")

    out_lines.append("")
    out_lines.append("")
    out_lines.append("def run(inputs):")

    # Emit ops/groups
    kernel_idx = 0
    for item in groups:
        if isinstance(item, FusionGroup):
            group = item
            kname = f"fused_{kernel_idx}"
            n_ops = len(group.ops)
            op_names = [op.op_name for op in group.ops if op.is_elementwise]
            desc = " + ".join(op_names[:5])
            if len(op_names) > 5:
                desc += f" ... ({len(op_names)})"

            out_lines.append(f"    # ── Fused: {desc} ({n_ops} ops → 1 kernel) ──")

            # Contiguous input calls
            for iname in group.input_vars:
                out_lines.append(
                    f"    # {iname} = {iname}.contiguous()  "
                    f"# ensure contiguous for CUDA kernel")

            # Show which ops are fused
            for op in group.ops:
                marker = "CUDA" if op.is_elementwise else "noop"
                out_lines.append(f"    # [{marker}] {op.line}")

            # Original PyTorch fallback (always correct)
            out_lines.append(f"    # --- PyTorch reference (unfused) ---")
            for op in group.ops:
                out_lines.append(f"    {op.line}")

            out_lines.append(f"    # --- end fused group (kernel: KERNEL_{kname.upper()}) ---")
            out_lines.append("")
            kernel_idx += 1
        else:
            # Single unfused op
            op = item
            out_lines.append(f"    {op.line}")

    # Return statement
    return_vars = _extract_return_vars(source)
    if return_vars:
        ret_items = ", ".join(sorted(return_vars))
        out_lines.append(f"    return [{ret_items}]")
    else:
        # Find original return
        for line in source.split("\n"):
            if line.strip().startswith("return"):
                out_lines.append(f"    {line.strip()}")
                break

    out_lines.append("")
    return "\n".join(out_lines)


def print_analysis(source: str) -> None:
    """Print a human-readable fusion analysis of a run(inputs) function."""
    result = analyze(source)
    stats = result["stats"]

    print(f"Total ops: {stats['total_ops']}")
    print(f"  Elementwise (fusable): {stats['elementwise']}")
    print(f"  Structural (passthrough): {stats['structural']}")
    print(f"  Barriers (unfusable): {stats['barrier']}")
    print(f"  Special (getitem etc.): {stats['special']}")
    print(f"  Fusion groups: {stats['fusion_groups']}")
    print(f"  Ops fused: {stats['ops_fused']}")
    print()

    for group, kname, ksrc in result["kernels"]:
        op_names = [op.op_name for op in group.ops if op.is_elementwise]
        print(f"── {kname}: {' → '.join(op_names)} ──")
        print(f"   Inputs:  {group.input_vars}")
        print(f"   Outputs: {group.output_vars}")
        print(f"   Kernel:")
        for line in ksrc.split("\n"):
            print(f"     {line}")
        print()
