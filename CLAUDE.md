# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

torch-graph captures PyTorch computation graphs at the aten op level and exports them as editable Python files. The primary workflow (`python -m torch_graph install`) replaces `torch.compile` transparently — models run using captured aten ops instead of Inductor, and users can edit the generated `.py` files to modify any operation.

## Setup

```bash
make setup    # Creates .venv, installs everything (PyTorch cu130, nanochat deps, etc.)
make test     # Run all tests
make all      # Tests + nanochat capture
```

Uses uv for dependency management. PyTorch 2.10+cu130. Requires CUDA GPU (H100/Hopper for FA3).

## Key architecture decisions

### Primal ordering
aot_autograd does NOT preserve `named_parameters()` order for FX placeholders. User inputs (idx, targets) can appear at ANY position. `capture.primal_names` records the true ordering via `data_ptr()` matching in `aot_backend`. None entries = user inputs, string entries = param/buffer paths.

### Install mechanism
Aten forward/backward are wrapped in `torch.autograd.Function` and monkey-patched onto the model. Parameters flow live (not frozen) — optimizer updates work. The install reads model params at runtime via `_make_live_attr_getter`.

### Optimizer capture and replay
Optimizer.step() is auto-captured on first call via patching `Optimizer.__init__`. With `replay_optimizer=True`, steps 2+ run through the captured aten graph instead of eager. Slot info maps each FX placeholder to its role (param/grad/state) using data_ptr matching. Lazily-initialized state (AdamW's exp_avg, exp_avg_sq, step) is enriched post-capture via `_enrich_unknown_slots`. Mutations are written back via `copy_()` under `torch.no_grad()`. The `_capture_depth` counter makes capture re-entrant for `@torch.compile`-decorated optimizer helpers (e.g. nanochat's `adamw_step_fused`).

### Inner compiled function capture and full replay (MuonAdamW etc.)
Optimizers like MuonAdamW use `@torch.compile` on inner helper functions (e.g. `adamw_step_fused`, `muon_step_fused`). These are detected via `_has_inner_compiled_fns()` which checks for `_CompiledFnProxy` instances in the optimizer's module. For these optimizers, we skip monolithic `optimizer.step()` capture and instead let each inner function's `_CompiledFnProxy` capture/replay its own aten graph independently.

With `replay_optimizer=True`, the system builds an `_InnerFnReplayPlan` on step 1 that completely replaces the original optimizer's outer loop on step 2+. The plan records each inner fn call's arg roles via `_match_arg_to_optimizer` (matching by data_ptr to param/grad/state/optimizer_attr, or by shape+value for stacked tensors). On replay, `_run_inner_replay` handles: (1) step counter increments, (2) per-call optimizer attr restoration (critical for muon LR prescaling which fills `_muon_lr_t` differently per group), (3) `torch.stack`/`torch._foreach_copy_` for muon param groups, and (4) calling each captured aten graph through its proxy. The original optimizer code is never called after step 1.

### Forward output layout
aot_autograd's forward returns a flat tuple: `(buffer_mutations..., real_outputs..., saved_for_backward...)`. The `num_mutations` count must be correct or BatchNorm models silently produce wrong results. See `_compute_num_mutations()` in `auto_install.py`.

### Backward saved tensor ordering
The aten backward function expects saved values in a specific order: non-tensor values (SymInts) first, then tensor values. This differs from the interleaved order returned by forward. See the reordering logic in `install.py` `_AtenGraph.backward()`.

### Multi-fragment support
Models with graph breaks (e.g. HF GPT2 with DynamicCache) produce 2+ forward/backward fragments. Backward graph ordering is REVERSED: `backward_0` corresponds to `forward_(n-1)`.

### Standalone training loop generation
`torch_graph/standalone.py` generates self-contained training scripts from captured aten graphs. After step 1 captures fw/bw/opt, `save_standalone_training()` saves initial params + optimizer state + a Python script that runs the full training loop using only aten files — no original model or optimizer code needed. Supports both monolithic optimizers (AdamW via slot_info) and inner-fn optimizers (MuonAdamW via serialized replay plan). Dynamic shapes are handled by resolving concrete SymInt values from sample inputs. CLI: `--verify N` generates and runs a standalone verification loop.

### Bit-identical capture
Captured aten graphs produce bit-identical forward outputs and gradients vs eager execution, including with Flash Attention 3 on Hopper GPUs. This is verified in tests.

## Footguns (things that will silently break if you get them wrong)

### SymInt concretization
Calling `int()` on a SymInt concretizes ALL related symbolic dimensions globally. In `export_aten_program`, forward/backward code must be generated BEFORE the weight-building loop that calls `int()` on SymInt placeholders. There is no way to undo concretization.

### FakeTensor detection
In PyTorch 2.10+, `untyped_storage()` succeeds on FakeTensors and `tolist()` produces symbolic `zuf0, zuf1, ...` names. Always use `_utils.is_fake(t)` before calling `.data_ptr()`, `.tolist()`, `.numpy()`, or pickle on tensors during tracing.

### Device fixup regex in multi-fragment export
The regex that adds `.to(_device)` to inline tensor constants in `_export_multi_fragment()` must ONLY match `torch.tensor/randn/zeros/ones/empty/full` — NOT aten ops like `torch.arange`. Source annotations in comments (e.g. `# Source: = torch.arange(`) span across newlines, and `[^)]*` will greedily match into the next line's `aten.arange(...)` call, producing broken code like `device=_device.to(_device)`.

### pytest fixture injection
Test functions that use pytest fixtures must NOT have default values on the fixture parameter. `def test_foo(fixture=None)` silently prevents pytest from injecting the fixture.

### nanochat recipe naming
Recipe file is `recipes/nanochat_wrapper.py` (not `nanochat.py`) to avoid name collision with the `nanochat` package when both are on sys.path.

## Working with the codebase

### Running tests
```bash
make test                                     # All tests
make test-quick                               # Fast (skip nanochat, NanoGPT)
.venv/bin/python -m pytest tests/ -v          # 121 pytest tests, ~80s
.venv/bin/python -m pytest tests/test_auto_install.py::test_single_linear -v  # Single test
.venv/bin/python -m pytest tests/ -v -k "optimizer"  # Tests matching keyword
.venv/bin/python run_tests.py                 # Tensor verification suite
.venv/bin/python test_models.py               # 80 model recipes, ~3 min (GPU)
.venv/bin/python test_models.py --only resnet18,hf_gpt2  # specific models
```

See `TESTS.md` for full test documentation: per-file test tables, coverage gaps, nanochat cache setup, and verification levels. Key test categories:
- **pytest** (`tests/`): 128+ tests — auto_install, install, multi-step, triton, H5, IR JSON, visualizer, inductor comparison, optimizer replay, MuonAdamW inner-fn capture, autoresearch E2E, nanochat E2E, standalone training loops, safety checks
- **run_tests.py**: Tensor verification (determinism, E2E accuracy, dynamic shapes, op dump)
- **test_models.py**: 80 model recipes (basic→HuggingFace) via `extract_training_step` + subprocess verification
- **test_fixes.py**: 52 legacy integration tests for historical fixes

### Key files to understand
- `torch_graph/auto_install.py` — The main workflow. Patches `torch.compile`, dispatches by variant (train/eval/arg pattern), caches to disk.
- `torch_graph/export.py` — `capture_aten_graphs()` + `export_aten_program()`. The graph capture and code generation core.
- `torch_graph/install.py` — `install()` function. Builds `torch.autograd.Function` from aten forward/backward, handles SymInt slots, buffer mutations.
- `torch_graph/_utils.py` — `RecordingInterpreter`, `is_fake()`, `materialize_tensor()`.
- `torch_graph/standalone.py` — `save_standalone_training()`. Generates self-contained training scripts from captured aten graphs + saved state.
- `torch_graph/ir_json.py` — Converts captured aten graphs to structured JSON IR (ops, args, kwargs, source maps).
- `torch_graph/condense_ir.py` — Condenses IR JSON by collapsing fused op groups and computing summary stats.
- `torch_graph/internal_ir.py` — Internal IR data structures shared by ir_json and condense_ir.
- `torch_graph/explain.py` — `explain()` one-liner API: capture + inspect + optional verify.
- `torch_graph/extract.py` — `extract_function()` / `extract_training_step()`: capture from arbitrary callables or training loops.
- `scripts/capture_nanochat.py` — Full nanochat capture: aten + HTML + H5 + kbox.

### Documentation
- `docs/EXTRACT_MODIFY_RUN.md` — How to extract aten graphs, edit them, and run with modifications (succinct + in-depth)
- `docs/WEAKNESSES.md` — All known weaknesses and limitations from stress testing

### Common patterns
- Clearing caches when debugging: `rm -rf .torch_graph_cache __pycache__`
- Nanochat capture: `make capture-nanochat` or `.venv/bin/python scripts/capture_nanochat.py`
- `torch._dynamo.reset()` between test runs that use different `torch.compile` configurations

### Environment
- Developed on Python 3.12.3, PyTorch 2.10.0+cu130, but also works with Python 3.10+ and PyTorch 2.9+
- uv for dependency management (if unavailable, `pip install -e ".[all]"` with system Python works)
- CUDA GPU required for nanochat tests (Flash Attention 3)
- Test repos in `outputs/repos/` (nanochat)
- NanoGPT test model in `test_repo/`

### Running without uv / without .venv
If `uv` is not installed, skip `make setup` and use system Python directly:
```bash
pip install torch==2.10.0+cu130 --index-url https://download.pytorch.org/whl/cu130
pip install -e ".[all]"
python3 -m pytest tests/ -v --tb=short         # run pytest
python3 run_tests.py --quick                   # tensor verification
```
The cu130 torch install is required — `pip install torch` alone gets the wrong CUDA version.

### Triton linker fix (libcuda.so)
If Triton tests or `torch.compile(backend="inductor")` fail with `/usr/bin/ld: cannot find -lcuda`, the CUDA stubs directory needs to be on `LIBRARY_PATH`:
```bash
export LIBRARY_PATH="/usr/local/cuda-13.1/targets/x86_64-linux/lib/stubs:${LIBRARY_PATH:-}"
```
This is needed when `libcuda.so` (unversioned symlink) doesn't exist but `libcuda.so.1` does. Add this to your shell profile or test runner.
