"""Generate standalone training loops from captured aten graphs.

After torch-graph captures forward/backward/optimizer aten graphs during step 1,
this module generates a self-contained training script that runs subsequent steps
using ONLY the captured aten files + saved initial state — no original model or
optimizer code needed.

Supports:
  - Monolithic optimizer capture (AdamW, SGD)
  - Inner fn capture (MuonAdamW with @torch.compile'd helpers)
  - Dynamic shapes (SymInt names in aten modules)

Usage:
    from torch_graph.standalone import save_standalone_training
    # After step 1 capture is complete:
    script_path = save_standalone_training(model, optimizer, cache_dir)
"""

from __future__ import annotations

import inspect
import json
import logging
import re
import textwrap
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn

logger = logging.getLogger("torch_graph")


def _resolve_attr(obj, path: str):
    """Resolve a dotted attribute path like 'layers.0.weight'."""
    for part in path.split('.'):
        obj = getattr(obj, part)
    return obj


def _read_meta(path: Path) -> dict:
    """Read .meta JSON alongside a .py aten file."""
    meta_path = path.with_suffix('.meta')
    if not meta_path.exists():
        return {}
    with open(meta_path) as f:
        return json.load(f)


def _load_aten_module(path: Path):
    """Import an aten .py file as a module, handling dynamic SymInt names."""
    import importlib.util
    source = path.read_text()

    # Pre-define any SymInt names (s0, s48, etc.) that appear in module-level
    # weight declarations. These are only needed for weight tensor shapes which
    # the standalone script never uses (it provides its own tensors).
    symint_names = set(re.findall(r'\b(s\d+)\b', source))

    spec = importlib.util.spec_from_file_location("aten_mod", str(path))
    mod = importlib.util.module_from_spec(spec)

    # Inject SymInt names as dummy integers before executing
    for name in symint_names:
        setattr(mod, name, 1)

    spec.loader.exec_module(mod)
    return mod


def _find_model_aten(cache_dir: Path) -> Path | None:
    """Find the model forward/backward aten .py file in cache_dir."""
    for py_file in sorted(cache_dir.glob("*_aten.py")):
        meta = _read_meta(py_file)
        if "model" in meta and "primal_names" in meta:
            return py_file
    for py_file in sorted(cache_dir.glob("*_aten.py")):
        meta = _read_meta(py_file)
        if "model" in meta:
            return py_file
    return None


def _find_optimizer_aten(cache_dir: Path) -> Path | None:
    """Find the monolithic optimizer aten .py file in cache_dir."""
    for py_file in sorted(cache_dir.glob("*_aten.py")):
        meta = _read_meta(py_file)
        if "optimizer_class" in meta and "slot_info" in meta:
            return py_file
    return None


# ---------------------------------------------------------------------------
# Inner fn replay plan serialization
# ---------------------------------------------------------------------------

def _get_inner_replay_plan(optimizer):
    """Get the inner fn replay plan from auto_install, if any."""
    try:
        from torch_graph.auto_install import _optimizer_captured
        entry = _optimizer_captured.get(id(optimizer), {})
        if entry.get("uses_inner_compiled"):
            return entry.get("inner_replay")
    except ImportError:
        pass
    return None


def _serialize_inner_fn_calls(
    plan,
    opt_to_primal: dict[tuple[int, int], int],
    optimizer: torch.optim.Optimizer,
    group_params: list[list[int]],
) -> tuple[list[dict], dict[int, torch.Tensor], list[int], float]:
    """Serialize inner fn replay plan into standalone-friendly format.

    Returns:
        calls: list of serialized call dicts
        state_tensors: {state_idx: tensor} for all state + optimizer_attr tensors
        step_state_indices: state indices that are step counters (per-param)
        initial_step: current step value from optimizer state or step attrs
    """
    state_tensors: dict[int, torch.Tensor] = {}
    state_idx = 0
    # Track state mapping: (gi, gpi, state_key) → state_idx
    state_map: dict[tuple, int] = {}
    # Track optimizer attr mapping: attr_name → state_idx
    attr_map: dict[str, int] = {}
    # Step counter state indices (per-param state["step"])
    step_state_indices: list[int] = []

    serialized_calls = []

    for call in plan.calls:
        proxy = call["proxy"]
        arg_roles = call["arg_roles"]

        # Find the correct variant for this call's input shapes.
        # Reconstruct call args from roles to build the dispatch key.
        call_args = []
        for role in arg_roles:
            r = role["role"]
            if r == "param":
                gi, gpi = role["group_index"], role["group_param_index"]
                call_args.append(optimizer.param_groups[gi]["params"][gpi])
            elif r == "grad":
                gi, gpi = role["group_index"], role["group_param_index"]
                param = optimizer.param_groups[gi]["params"][gpi]
                call_args.append(
                    param.grad if param.grad is not None
                    else torch.zeros_like(param)
                )
            elif r == "state":
                gi, gpi = role["group_index"], role["group_param_index"]
                param = optimizer.param_groups[gi]["params"][gpi]
                call_args.append(optimizer.state[param][role["state_key"]])
            elif r == "optimizer_attr":
                call_args.append(getattr(optimizer, role["attr_name"]))
            elif r == "stacked_params":
                gi = role["group_index"]
                call_args.append(
                    torch.stack(list(optimizer.param_groups[gi]["params"]))
                )
            elif r == "stacked_grads":
                gi = role["group_index"]
                params = optimizer.param_groups[gi]["params"]
                call_args.append(
                    torch.stack([p.grad if p.grad is not None
                                 else torch.zeros_like(p) for p in params])
                )
            elif r == "constant":
                call_args.append(role["value"])
            else:
                call_args.append(None)

        # Match variant via dispatch key
        key = proxy._dispatch_key(tuple(call_args), {})
        variant = proxy._variants.get(key)

        # Fallback: try first available variant
        if variant is None:
            for v in proxy._variants.values():
                if v is not None and v.cache_path is not None:
                    variant = v
                    break

        if variant is None:
            logger.warning(
                f"Inner fn proxy {proxy._fn_name()} has no cached variant, skipping"
            )
            continue

        # Serialize arg_roles with fw_pos and state_idx references
        serialized_roles = []
        for role in arg_roles:
            r = role["role"]
            if r == "param":
                gi, gpi = role["group_index"], role["group_param_index"]
                fw_pos = opt_to_primal.get((gi, gpi), -1)
                serialized_roles.append({"role": "param", "fw_pos": fw_pos})

            elif r == "grad":
                gi, gpi = role["group_index"], role["group_param_index"]
                fw_pos = opt_to_primal.get((gi, gpi), -1)
                serialized_roles.append({"role": "grad", "fw_pos": fw_pos})

            elif r == "state":
                gi = role["group_index"]
                gpi = role["group_param_index"]
                sk = role["state_key"]
                key = (gi, gpi, sk)
                if key not in state_map:
                    param = optimizer.param_groups[gi]["params"][gpi]
                    val = optimizer.state[param].get(sk)
                    if isinstance(val, torch.Tensor):
                        state_tensors[state_idx] = val.detach().clone()
                    elif val is not None:
                        state_tensors[state_idx] = torch.tensor(
                            float(val), device=param.device
                        )
                    else:
                        state_tensors[state_idx] = torch.tensor(
                            0.0, device=param.device
                        )
                    state_map[key] = state_idx
                    # Track step counters
                    if sk == "step":
                        step_state_indices.append(state_idx)
                    state_idx += 1
                sr = {"role": "state", "state_idx": state_map[key]}
                serialized_roles.append(sr)

            elif r == "optimizer_attr":
                attr_name = role["attr_name"]
                if attr_name not in attr_map:
                    val = getattr(optimizer, attr_name, None)
                    device = next(iter(
                        p.device for g in optimizer.param_groups
                        for p in g["params"]
                    ))
                    if isinstance(val, torch.Tensor):
                        state_tensors[state_idx] = val.detach().clone()
                    elif val is not None:
                        state_tensors[state_idx] = torch.tensor(
                            float(val), device=device
                        )
                    else:
                        state_tensors[state_idx] = torch.tensor(0.0, device=device)
                    attr_map[attr_name] = state_idx
                    state_idx += 1
                sr = {
                    "role": "optimizer_attr",
                    "state_idx": attr_map[attr_name],
                    "is_step_attr": attr_name in plan.step_attr_names,
                }
                if "captured_value" in role:
                    sr["captured_value"] = role["captured_value"]
                serialized_roles.append(sr)

            elif r == "stacked_params":
                gi = role["group_index"]
                serialized_roles.append({
                    "role": "stacked_params",
                    "group_index": gi,
                })

            elif r == "stacked_grads":
                gi = role["group_index"]
                serialized_roles.append({
                    "role": "stacked_grads",
                    "group_index": gi,
                })

            elif r == "constant":
                serialized_roles.append({
                    "role": "constant",
                    "value": role["value"],
                })

            else:
                serialized_roles.append({"role": "unknown"})

        fn_file = Path(variant.cache_path).name
        serialized_calls.append({
            "fn_file": fn_file,
            "num_mutations": variant.num_mutations,
            "mutated_arg_indices": list(variant.mutated_arg_indices),
            "call_order": variant.get_call_order(),
            "symint_specs": [list(s) for s in variant.symint_specs],
            "arg_roles": serialized_roles,
            "copy_back_groups": list(call["copy_back_groups"]),
        })

    # Determine initial step value.  Sources (in priority order):
    # 1. Per-param state["step"] (already tracked in step_state_indices)
    # 2. Optimizer attr with is_step_attr=True (captured_value from step 1)
    # 3. Fallback: 1.0 (default after 1 training step)
    initial_step = 1.0
    if step_state_indices:
        initial_step = float(state_tensors[step_state_indices[0]].item())
    else:
        # No per-param step counters — check optimizer attrs
        for group in optimizer.param_groups:
            for param in group["params"]:
                st = optimizer.state.get(param, {})
                if "step" in st:
                    val = st["step"]
                    initial_step = float(
                        val.item() if isinstance(val, torch.Tensor) else val
                    )
                    break
            else:
                continue
            break

    return serialized_calls, state_tensors, step_state_indices, initial_step


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def save_standalone_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cache_dir: str | Path,
    *,
    model_aten_path: str | Path | None = None,
    opt_aten_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    num_steps: int = 5,
    sample_inputs: tuple[torch.Tensor, ...] | None = None,
) -> Path:
    """Generate a standalone training script from captured aten graphs.

    Call after the first training step when fw+bw+opt captures are complete.
    The generated script runs the full training loop using only the captured
    aten files + saved initial state — no original model or optimizer needed.

    Supports both monolithic optimizer capture (AdamW, SGD) and inner fn
    capture (MuonAdamW with @torch.compile'd helpers like adamw_step_fused).

    Args:
        model: The model (for reading live param/buffer values)
        optimizer: The optimizer (for reading live state values)
        cache_dir: Directory containing captured aten .py/.meta files
        model_aten_path: Path to model fw/bw aten .py (auto-detected if None)
        opt_aten_path: Path to monolithic optimizer aten .py (auto-detected if None)
        output_dir: Where to write the script + state (defaults to cache_dir)
        num_steps: Default number of training steps in generated script
        sample_inputs: User input tensors (x, target, etc.) to save for
            reproducible runs. If None, the generated script creates random data.

    Returns:
        Path to the generated standalone_training.py script
    """
    cache_dir = Path(cache_dir)
    output_dir = Path(output_dir or cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect aten file paths
    if model_aten_path is None:
        model_aten_path = _find_model_aten(cache_dir)
        if model_aten_path is None:
            raise FileNotFoundError(
                f"No model aten .py found in {cache_dir}. "
                "Run a training step with auto_install first."
            )
    model_aten_path = Path(model_aten_path)

    # Read model metadata
    model_meta = _read_meta(model_aten_path)
    primal_names = model_meta.get("primal_names", [])
    num_mutations = model_meta.get("num_mutations", 0)
    num_real_outputs = model_meta.get("num_real_outputs", 1)

    if not primal_names:
        raise ValueError(
            f"Model .meta at {model_aten_path} has no primal_names. "
            "Recapture with the latest auto_install."
        )

    # Classify forward input positions.
    # None entries in primal_names can be either tensor user inputs (x, target)
    # or SymInt scalars (batch size). We distinguish them via the forward
    # signature: SymInts have no type annotation, tensors have one.
    param_set = set(dict(model.named_parameters()).keys())
    buffer_set = set(dict(model.named_buffers()).keys())

    # Load forward signature to classify None positions
    aten_mod = _load_aten_module(model_aten_path)
    fw_sig = inspect.signature(aten_mod.forward)
    fw_param_list = list(fw_sig.parameters.values())

    param_positions = []
    buffer_positions = []
    user_positions = []     # real tensor user inputs (x, target, etc.)
    symint_positions = []   # SymInt scalar positions (batch size, etc.)
    n_fw = len(primal_names)

    for i, name in enumerate(primal_names):
        if name is None:
            # Check forward signature for annotation
            if i < len(fw_param_list):
                ann = fw_param_list[i].annotation
                if ann == inspect.Parameter.empty:
                    symint_positions.append(i)
                else:
                    user_positions.append(i)
            else:
                user_positions.append(i)
        elif name in param_set:
            param_positions.append(i)
        elif name in buffer_set:
            buffer_positions.append(i)
        else:
            user_positions.append(i)

    # Build fw_pos <-> optimizer (gi, gpi) mapping via data_ptr
    primal_ptr_map: dict[int, int] = {}
    for i, name in enumerate(primal_names):
        if name is not None:
            try:
                param = _resolve_attr(model, name)
                if isinstance(param, torch.Tensor):
                    primal_ptr_map[param.data_ptr()] = i
            except Exception:
                pass

    opt_to_primal: dict[tuple[int, int], int] = {}
    group_params: list[list[int]] = []
    for gi, group in enumerate(optimizer.param_groups):
        gp = []
        for gpi, p in enumerate(group["params"]):
            fw_pos = primal_ptr_map.get(p.data_ptr())
            if fw_pos is not None:
                opt_to_primal[(gi, gpi)] = fw_pos
                gp.append(fw_pos)
        group_params.append(gp)

    # Check for inner fn replay plan (MuonAdamW etc.)
    inner_replay = _get_inner_replay_plan(optimizer)
    use_inner_fn = inner_replay is not None

    # Build optimizer state
    opt_state_tensors: dict[int, torch.Tensor] = {}
    opt_slots: list[str] = []
    opt_mutated_slots: list[int] = []
    inner_fn_calls: list[dict] = []
    step_state_indices: list[int] = []

    initial_step = 1.0
    if use_inner_fn:
        inner_fn_calls, opt_state_tensors, step_state_indices, initial_step = (
            _serialize_inner_fn_calls(
                inner_replay, opt_to_primal, optimizer, group_params,
            )
        )
    else:
        # Monolithic optimizer
        if opt_aten_path is None:
            opt_aten_path = _find_optimizer_aten(cache_dir)
        if opt_aten_path is not None:
            opt_aten_path = Path(opt_aten_path)

        if opt_aten_path is not None:
            opt_meta = _read_meta(opt_aten_path)
            slot_info = opt_meta.get("slot_info", [])
            opt_mutated_slots = opt_meta.get("mutated_slot_indices", [])

            state_idx = 0
            for info in slot_info:
                role = info.get("role", "unknown")
                gi = info.get("group_index")
                gpi = info.get("group_param_index")

                if role == "param":
                    fw_pos = opt_to_primal.get((gi, gpi))
                    opt_slots.append(f"p:{fw_pos if fw_pos is not None else -1}")
                elif role == "grad":
                    fw_pos = opt_to_primal.get((gi, gpi))
                    opt_slots.append(f"g:{fw_pos if fw_pos is not None else -1}")
                elif role == "state":
                    state_key = info.get("state_key", "unknown")
                    param = optimizer.param_groups[gi]["params"][gpi]
                    val = optimizer.state[param].get(state_key)
                    if isinstance(val, torch.Tensor):
                        opt_state_tensors[state_idx] = val.detach().clone()
                    elif val is not None:
                        opt_state_tensors[state_idx] = torch.tensor(
                            float(val), device=param.device
                        )
                    else:
                        opt_state_tensors[state_idx] = torch.tensor(
                            0.0, device=param.device
                        )
                    opt_slots.append(f"s:{state_idx}")
                    state_idx += 1
                elif role == "optimizer_attr":
                    attr_name = info.get("attr_name", "")
                    val = getattr(optimizer, attr_name, None)
                    device = next(model.parameters()).device
                    if isinstance(val, torch.Tensor):
                        opt_state_tensors[state_idx] = val.detach().clone()
                    elif val is not None:
                        opt_state_tensors[state_idx] = torch.tensor(
                            float(val), device=device
                        )
                    else:
                        opt_state_tensors[state_idx] = torch.tensor(
                            0.0, device=device
                        )
                    opt_slots.append(f"s:{state_idx}")
                    state_idx += 1
                else:
                    device = next(model.parameters()).device
                    opt_state_tensors[state_idx] = torch.tensor(0.0, device=device)
                    opt_slots.append(f"s:{state_idx}")
                    state_idx += 1

    # Save initial param/buffer values (keyed by fw_pos)
    fw_params: dict[int, torch.Tensor] = {}
    for i, name in enumerate(primal_names):
        if name is None:
            continue
        try:
            obj = _resolve_attr(model, name)
            if isinstance(obj, torch.Tensor):
                fw_params[i] = obj.detach().clone()
        except Exception:
            pass

    # Compute SymInt concrete values from sample inputs.
    # SymInt positions typically hold batch-size-like values; derive from
    # the first sample input's shape if available.
    symint_values: dict[int, int] = {}
    if sample_inputs and symint_positions:
        # Heuristic: the SymInt is the batch dimension (shape[0]) of the
        # first tensor sample input.
        for si_pos in symint_positions:
            for inp in sample_inputs:
                if isinstance(inp, torch.Tensor) and inp.dim() > 0:
                    symint_values[si_pos] = inp.shape[0]
                    break
            if si_pos not in symint_values:
                symint_values[si_pos] = 1  # fallback

    # Save state
    state_path = output_dir / "standalone_state.pt"
    state_dict: dict[str, Any] = {
        "params": fw_params,
        "opt_state": opt_state_tensors,
    }
    if sample_inputs is not None:
        state_dict["sample_inputs"] = [
            t.detach().clone() if isinstance(t, torch.Tensor) else t
            for t in sample_inputs
        ]
    if use_inner_fn:
        state_dict["inner_fn_calls"] = inner_fn_calls
        state_dict["group_params"] = group_params
        state_dict["step_state_indices"] = step_state_indices
        state_dict["initial_step"] = initial_step
    torch.save(state_dict, state_path)

    # Get user input info from forward signature (aten_mod already loaded above)
    fw_param_names = list(fw_sig.parameters.keys())
    user_input_info = []
    for pos in user_positions:
        if pos < len(fw_param_names):
            pname = fw_param_names[pos]
            ann = fw_sig.parameters[pname].annotation
            ann_str = str(ann) if ann != inspect.Parameter.empty else None
            user_input_info.append({
                "position": pos,
                "name": pname,
                "annotation": ann_str,
            })

    # Build forward input layout comment
    layout_lines = []
    for i, name in enumerate(primal_names):
        if name is None:
            ui = next((u for u in user_input_info if u["position"] == i), None)
            if ui:
                ann = f" {ui['annotation']}" if ui.get("annotation") else ""
                layout_lines.append(f"#   [{i}] user_input \"{ui['name']}\"{ann}")
            else:
                layout_lines.append(f"#   [{i}] user_input")
        elif name in param_set:
            shape = list(fw_params[i].shape) if i in fw_params else "?"
            layout_lines.append(f"#   [{i}] param \"{name}\" {shape}")
        elif name in buffer_set:
            shape = list(fw_params[i].shape) if i in fw_params else "?"
            layout_lines.append(f"#   [{i}] buffer \"{name}\" {shape}")

    # Generate the script
    if use_inner_fn:
        script = _generate_inner_fn_script(
            model_aten_path=model_aten_path,
            state_path=state_path,
            n_fw=n_fw,
            num_mutations=num_mutations,
            num_real_outputs=num_real_outputs,
            param_positions=param_positions,
            buffer_positions=buffer_positions,
            user_positions=user_positions,
            symint_positions=symint_positions,
            symint_values=symint_values,
            user_input_info=user_input_info,
            layout_lines=layout_lines,
            inner_fn_calls=inner_fn_calls,
            group_params=group_params,
            step_state_indices=step_state_indices,
            initial_step=initial_step,
            num_steps=num_steps,
            model_name=model_meta.get("model", "Model"),
            has_sample_inputs=sample_inputs is not None,
        )
    else:
        script = _generate_monolithic_script(
            model_aten_path=model_aten_path,
            opt_aten_path=opt_aten_path,
            state_path=state_path,
            n_fw=n_fw,
            num_mutations=num_mutations,
            num_real_outputs=num_real_outputs,
            param_positions=param_positions,
            buffer_positions=buffer_positions,
            user_positions=user_positions,
            symint_positions=symint_positions,
            symint_values=symint_values,
            user_input_info=user_input_info,
            layout_lines=layout_lines,
            opt_slots=opt_slots,
            opt_mutated_slots=opt_mutated_slots,
            num_steps=num_steps,
            model_name=model_meta.get("model", "Model"),
            opt_name=(opt_aten_path.stem if opt_aten_path else "None"),
            has_sample_inputs=sample_inputs is not None,
        )

    script_path = output_dir / "standalone_training.py"
    script_path.write_text(script)
    logger.info(f"Generated standalone training script: {script_path}")
    return script_path


# ---------------------------------------------------------------------------
# Script generation helpers
# ---------------------------------------------------------------------------

def _build_data_section(has_sample_inputs: bool, user_input_info: list[dict]) -> str:
    """Build the data generation code for the training loop."""
    if has_sample_inputs:
        return '    data = S["sample_inputs"]'

    data_gen_lines = []
    for ui in user_input_info:
        ann = ui.get("annotation", "")
        name = ui["name"]
        if ann and "int" in ann.lower():
            m = re.match(r"'?\w+\[([^\]]+)\]'?", ann or "")
            if m:
                shape = f"[{m.group(1)}]"
                data_gen_lines.append(
                    f"    {name} = torch.randint(0, 64, {shape}, device=DEVICE)"
                )
            else:
                data_gen_lines.append(
                    f"    {name} = torch.randint(0, 64, (16,), device=DEVICE)"
                )
        elif ann:
            m = re.match(r"'?\w+\[([^\]]+)\]'?", ann or "")
            if m:
                shape = f"[{m.group(1)}]"
                data_gen_lines.append(
                    f"    {name} = torch.randn({shape}, device=DEVICE)"
                )
            else:
                data_gen_lines.append(
                    f"    {name} = torch.randn(16, device=DEVICE)"
                )
        else:
            data_gen_lines.append(
                f"    {name} = torch.randn(16, device=DEVICE)  # TODO: set shape"
            )
    data_names = [ui["name"] for ui in user_input_info]
    data_list = ", ".join(data_names)
    if data_gen_lines:
        return "\n".join(data_gen_lines) + f"\n    data = [{data_list}]"
    return "    data = []"


def _build_user_insert(user_input_info: list[dict]) -> str:
    """Build the user input insertion code."""
    lines = []
    for i, ui in enumerate(user_input_info):
        lines.append(f"    fw_in[{ui['position']}] = data[{i}]")
    return "\n".join(lines)


def _build_loader_fn() -> str:
    """Build the _load function with SymInt support."""
    return (
        "def _load(filename):\n"
        '    """Load an aten .py module from CACHE directory, handling dynamic SymInt names."""\n'
        "    import re as _re\n"
        "    p = CACHE / filename\n"
        "    source = p.read_text()\n"
        "    symint_names = set(_re.findall(r'\\b(s\\d+)\\b', source))\n"
        "    spec = importlib.util.spec_from_file_location('m', str(p))\n"
        "    mod = importlib.util.module_from_spec(spec)\n"
        "    for name in symint_names:\n"
        "        setattr(mod, name, 1)  # dummy value — module-level weights unused\n"
        "    spec.loader.exec_module(mod)\n"
        "    return mod\n"
    )


def _build_header(model_aten_path: Path, state_path: Path, model_name: str,
                   num_steps: int) -> str:
    """Build the common script header."""
    lines = [
        f'"""Standalone training loop for {model_name}.',
        'Generated by torch-graph. Edit freely — no original model/optimizer code needed.',
        '"""',
        'import torch',
        'import importlib.util',
        'from pathlib import Path',
        '',
        '',
    ]
    header = "\n".join(lines) + "\n"
    header += _build_loader_fn()
    header += "\n"
    header += "\n".join([
        f'# ── Configuration ──',
        f'CACHE = Path("{model_aten_path.parent}")',
        f'NUM_STEPS = {num_steps}',
        f'DEVICE = "cuda" if torch.cuda.is_available() else "cpu"',
        f'',
        f'# ── Load state ──',
        f'S = torch.load(CACHE / "{state_path.name}", weights_only=False, map_location=DEVICE)',
        f'params = S["params"]       # {{fw_pos: tensor}} — live params updated by optimizer',
        f'opt_state = S["opt_state"] # {{idx: tensor}} — optimizer state tensors',
        '',
    ])
    return header


# ---------------------------------------------------------------------------
# Monolithic optimizer script
# ---------------------------------------------------------------------------

def _generate_monolithic_script(
    *,
    model_aten_path: Path,
    opt_aten_path: Path | None,
    state_path: Path,
    n_fw: int,
    num_mutations: int,
    num_real_outputs: int,
    param_positions: list[int],
    buffer_positions: list[int],
    user_positions: list[int],
    symint_positions: list[int],
    symint_values: dict[int, int],
    user_input_info: list[dict],
    layout_lines: list[str],
    opt_slots: list[str],
    opt_mutated_slots: list[int],
    num_steps: int,
    model_name: str,
    opt_name: str,
    has_sample_inputs: bool = False,
) -> str:
    """Generate standalone script for monolithic optimizer (AdamW, SGD)."""

    data_gen_code = _build_data_section(has_sample_inputs, user_input_info)
    user_insert_code = _build_user_insert(user_input_info)
    layout_comment = "\n".join(layout_lines)

    if opt_aten_path is not None and opt_slots:
        opt_load = f'opt = _load("{opt_aten_path.name}")'
        opt_slots_str = repr(opt_slots)
        opt_mut_str = repr(opt_mutated_slots)
        opt_section = (
            "# === OPTIMIZER ===\n"
            "    oi = []\n"
            "    for slot in OPT_SLOTS:\n"
            "        kind, key = slot.split(':')\n"
            "        key = int(key)\n"
            "        if kind == 'p': oi.append(params[key])\n"
            "        elif kind == 'g': oi.append(grads[key])\n"
            "        elif kind == 's': oi.append(opt_state[key])\n"
            "\n"
            "    with torch.no_grad():\n"
            "        r = opt.forward(*oi)\n"
            "        r = r if isinstance(r, tuple) else (r,)\n"
            "        for out_i, slot_i in enumerate(OPT_MUT):\n"
            "            oi[slot_i].copy_(r[out_i])"
        )
    else:
        opt_load = "opt = None  # No optimizer aten captured"
        opt_slots_str = "[]"
        opt_mut_str = "[]"
        opt_section = "    # No optimizer aten — use torch.optim directly if needed"

    header = _build_header(model_aten_path, state_path, model_name, num_steps)

    body_lines = [
        '',
        f'# ── Load aten modules ──',
        f'fw_bw = _load("{model_aten_path.name}")',
        opt_load,
        '',
        f'# ── Forward layout ({n_fw} inputs) ──',
        layout_comment,
        f'N_FW = {n_fw}',
        f'N_MUT = {num_mutations}       # buffer mutations in forward output',
        f'N_OUT = {num_real_outputs}       # real outputs (loss)',
        f'PARAM_POS = {param_positions}',
        f'BUF_POS = {buffer_positions}',
        f'USER_POS = {user_positions}',
        '',
        '# ── Optimizer layout ──',
        '# "p:N" = param at fw pos N, "g:N" = grad at fw pos N, "s:N" = opt_state[N]',
        f'OPT_SLOTS = {opt_slots_str}',
        f'OPT_MUT = {opt_mut_str}',
        '',
        '# ── Training loop ──',
        'losses = []',
        'for step in range(NUM_STEPS):',
        '    # === DATA ===',
        data_gen_code,
        '',
        '    # === FORWARD ===',
        '    fw_in = [None] * N_FW',
        '    for p, t in params.items(): fw_in[p] = t',
        user_insert_code,
    ]
    for pos, val in sorted(symint_values.items()):
        body_lines.append(f'    fw_in[{pos}] = {val}  # SymInt (batch size)')
    body_lines += [
        '',
        '    result = fw_bw.forward(*fw_in)',
        '    loss = result[N_MUT]',
        '    saved = list(result[N_MUT + N_OUT:])',
        '    # Reorder saved: backward expects (non-tensors..., tensors..., tangents)',
        '    non_t = [v for v in saved if not isinstance(v, torch.Tensor)]',
        '    tens = [v for v in saved if isinstance(v, torch.Tensor)]',
        '    saved = non_t + tens if non_t else saved',
        '',
        '    # === BACKWARD ===',
        '    grads = fw_bw.backward(*saved, torch.ones_like(loss))',
        '',
        '    ' + opt_section,
        '',
        '    losses.append(loss.item())',
        '    print(f"Step {step}: loss={loss.item():.6f}")',
        '',
        'print(f"\\nAll losses: {losses}")',
        '',
    ]
    return header + "\n".join(body_lines)


# ---------------------------------------------------------------------------
# Inner fn optimizer script (MuonAdamW etc.)
# ---------------------------------------------------------------------------

def _generate_inner_fn_script(
    *,
    model_aten_path: Path,
    state_path: Path,
    n_fw: int,
    num_mutations: int,
    num_real_outputs: int,
    param_positions: list[int],
    buffer_positions: list[int],
    user_positions: list[int],
    symint_positions: list[int],
    symint_values: dict[int, int],
    user_input_info: list[dict],
    layout_lines: list[str],
    inner_fn_calls: list[dict],
    group_params: list[list[int]],
    step_state_indices: list[int],
    initial_step: float = 1.0,
    num_steps: int = 5,
    model_name: str = "Model",
    has_sample_inputs: bool = False,
) -> str:
    """Generate standalone script for inner fn optimizer (MuonAdamW etc.)."""

    data_gen_code = _build_data_section(has_sample_inputs, user_input_info)
    user_insert_code = _build_user_insert(user_input_info)
    layout_comment = "\n".join(layout_lines)

    # Collect unique inner fn files
    fn_files = []
    fn_file_to_idx: dict[str, int] = {}
    for call in inner_fn_calls:
        fn = call["fn_file"]
        if fn not in fn_file_to_idx:
            fn_file_to_idx[fn] = len(fn_files)
            fn_files.append(fn)

    fn_load_lines = []
    for i, fn in enumerate(fn_files):
        fn_load_lines.append(f'inner_fns.append(_load("{fn}"))  # [{i}] {fn}')
    fn_loads = "\n".join(fn_load_lines)

    header = _build_header(model_aten_path, state_path, model_name, num_steps)

    # Build the call dispatch function inline
    # Each call: assemble args from roles, reorder via call_order, call forward,
    # write mutations, copy back stacked params
    call_code_parts = []
    for ci, call in enumerate(inner_fn_calls):
        fn_idx = fn_file_to_idx[call["fn_file"]]
        roles = call["arg_roles"]
        n_mut = call["num_mutations"]
        mut_indices = call["mutated_arg_indices"]
        call_order = call["call_order"]
        symint_specs = call["symint_specs"]
        copy_back = call["copy_back_groups"]

        lines = [f"        # Call {ci}: {call['fn_file']}"]

        # Restore per-call optimizer attr values
        for ri, role in enumerate(roles):
            if role["role"] == "optimizer_attr" and "captured_value" in role:
                si = role["state_idx"]
                cv = role["captured_value"]
                lines.append(f"        opt_state[{si}].fill_({cv!r})")

        # Override step attrs with new_step
        for ri, role in enumerate(roles):
            if (role["role"] == "optimizer_attr"
                    and role.get("is_step_attr")):
                si = role["state_idx"]
                lines.append(f"        opt_state[{si}].fill_(new_step)")

        # Assemble args
        lines.append(f"        args_{ci} = []")
        for ri, role in enumerate(roles):
            r = role["role"]
            if r == "param":
                lines.append(f"        args_{ci}.append(params[{role['fw_pos']}])")
            elif r == "grad":
                lines.append(f"        args_{ci}.append(grads[{role['fw_pos']}])")
            elif r == "state":
                lines.append(f"        args_{ci}.append(opt_state[{role['state_idx']}])")
            elif r == "optimizer_attr":
                lines.append(f"        args_{ci}.append(opt_state[{role['state_idx']}])")
            elif r == "stacked_params":
                gi = role["group_index"]
                gp = group_params[gi]
                lines.append(f"        args_{ci}.append(torch.stack([params[p] for p in {gp}]))")
            elif r == "stacked_grads":
                gi = role["group_index"]
                gp = group_params[gi]
                lines.append(f"        args_{ci}.append(torch.stack([grads[p] for p in {gp}]))")
            elif r == "constant":
                lines.append(f"        args_{ci}.append({role['value']!r})")
            else:
                lines.append(f"        args_{ci}.append(None)  # unknown role")

        # Build FX inputs (reorder via call_order + symint specs)
        lines.append(f"        fx_in_{ci} = []")
        for ca, dim in symint_specs:
            lines.append(f"        fx_in_{ci}.append(args_{ci}[{ca}].shape[{dim}])")
        for idx in call_order:
            lines.append(f"        fx_in_{ci}.append(args_{ci}[{idx}])")

        # Call forward
        lines.append(f"        result_{ci} = inner_fns[{fn_idx}].forward(*fx_in_{ci})")

        # Write mutations back
        if n_mut > 0:
            lines.append(f"        r_{ci} = result_{ci} if isinstance(result_{ci}, (tuple, list)) else (result_{ci},)")
            for out_i, arg_i in enumerate(mut_indices):
                lines.append(f"        args_{ci}[{arg_i}].copy_(r_{ci}[{out_i}])")

        # Copy back stacked params
        for gi in copy_back:
            gp = group_params[gi]
            # Find the arg index for stacked_params of this group
            for ai, role in enumerate(roles):
                if role["role"] == "stacked_params" and role["group_index"] == gi:
                    lines.append(f"        torch._foreach_copy_([params[p] for p in {gp}], list(args_{ci}[{ai}].unbind(0)))")
                    break

        call_code_parts.append("\n".join(lines))

    all_call_code = "\n\n".join(call_code_parts)

    # Build inner fn file list for the comment
    fn_list_comment = "\n".join(
        f"#   [{i}] {fn}" for i, fn in enumerate(fn_files)
    )

    body_lines = [
        '',
        f'# ── Load aten modules ──',
        f'fw_bw = _load("{model_aten_path.name}")',
        'inner_fns = []',
        fn_loads,
        '',
        f'# ── Forward layout ({n_fw} inputs) ──',
        layout_comment,
        f'N_FW = {n_fw}',
        f'N_MUT = {num_mutations}       # buffer mutations in forward output',
        f'N_OUT = {num_real_outputs}       # real outputs (loss)',
        f'PARAM_POS = {param_positions}',
        f'BUF_POS = {buffer_positions}',
        f'USER_POS = {user_positions}',
        '',
        f'# ── Inner fn optimizer layout ──',
        f'# {len(inner_fn_calls)} calls across {len(fn_files)} captured functions:',
        fn_list_comment,
        f'GROUP_PARAMS = {group_params}',
        f'STEP_STATE = {step_state_indices}   # opt_state indices that are step counters',
        f'INITIAL_STEP = {initial_step}  # step counter value after capture',
        '',
        '# ── Training loop ──',
        'losses = []',
        '_step_counter = INITIAL_STEP  # tracks optimizer step number',
        'for step in range(NUM_STEPS):',
        '    # === DATA ===',
        data_gen_code,
        '',
        '    # === FORWARD ===',
        '    fw_in = [None] * N_FW',
        '    for p, t in params.items(): fw_in[p] = t',
        user_insert_code,
    ]
    for pos, val in sorted(symint_values.items()):
        body_lines.append(f'    fw_in[{pos}] = {val}  # SymInt (batch size)')
    body_lines += [
        '',
        '    result = fw_bw.forward(*fw_in)',
        '    loss = result[N_MUT]',
        '    saved = list(result[N_MUT + N_OUT:])',
        '    # Reorder saved: backward expects (non-tensors..., tensors..., tangents)',
        '    non_t = [v for v in saved if not isinstance(v, torch.Tensor)]',
        '    tens = [v for v in saved if isinstance(v, torch.Tensor)]',
        '    saved = non_t + tens if non_t else saved',
        '',
        '    # === BACKWARD ===',
        '    grads = fw_bw.backward(*saved, torch.ones_like(loss))',
        '',
        '    # === OPTIMIZER (inner fn replay) ===',
        '    # Increment step counter',
        '    _step_counter += 1',
        '    new_step = _step_counter',
        '    for si in STEP_STATE:',
        '        opt_state[si] = opt_state[si] + 1',
        '',
        '    with torch.no_grad():',
        all_call_code,
        '',
        '    losses.append(loss.item())',
        '    print(f"Step {step}: loss={loss.item():.6f}")',
        '',
        'print(f"\\nAll losses: {losses}")',
        '',
    ]
    return header + "\n".join(body_lines)
