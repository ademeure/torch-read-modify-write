# Test Documentation

## Environment

- **Setup:** Run `make setup` once to create `.venv` and install all dependencies (PyTorch cu130, nanochat, etc.)
- **Python:** 3.12, managed by `uv` — always use `.venv/bin/python`, not system Python
- **GPU:** CUDA required for most tests; H100/Hopper required for Flash Attention 3 (nanochat tests)
- **CPU-only:** pytest tests skip CUDA-dependent tests automatically; `run_tests.py` requires GPU

## Quick Reference

| Command | What it runs | Time | GPU? |
|---------|-------------|------|------|
| `make test` | pytest + run_tests.py | ~30s + ~15s | Yes (some skip on CPU) |
| `make test-quick` | pytest (no nanochat) + run_tests.py --quick | ~20s + ~5s | Yes (some skip on CPU) |
| `make test-models` | test_models.py (80 model recipes) | ~3 min | Yes |
| `.venv/bin/python test_fixes.py` | Legacy integration tests | ~15s | Yes |
| `.venv/bin/python test.py` | Example script (HTML output) | ~5s | Yes |

**Recommended workflow:** `make test` for normal development. Run `make test-models` before merging major changes.

---

## 1. pytest tests (`tests/`)

Run with: `.venv/bin/python -m pytest tests/ -v --tb=short`

**Current status: 121 passed, 0 failed, 0 skipped** (verified on 2026-03-13)

### tests/test_auto_install.py (44 tests)

The core test file. Tests the `torch.compile` interception and aten op replacement pipeline.

| Test | What it verifies |
|------|-----------------|
| `test_single_linear` | Single FC layer capture + install |
| `test_tiny_mlp` | Multi-layer MLP with ReLU |
| `test_batchnorm` | MLP + BatchNorm (buffer mutations) |
| `test_conv_model` | Conv2d + Pool + FC |
| `test_model_with_kwargs` | Model with idx/targets kwargs (LM-style) |
| `test_multi_output` | Model with 2 output heads |
| `test_dropout` | Dropout in training mode |
| `test_residual_network` | Residual blocks + LayerNorm |
| `test_transformer` | Mini transformer encoder |
| `test_training_loop` | Multiple optimizer steps |
| `test_cache_reload` | Loading from cached .py files |
| `test_force_recapture` | force_recapture flag |
| `test_force_recapture_eval_variant` | force_recapture also refreshes eval-only cache files |
| `test_h5_dump_creates_file` | H5 output generation |
| `test_user_modified_model_backward_changes_gradients` | Edit backward() in cache affects grads |
| `test_user_modified_optimizer_function_changes_updates` | Edit optimizer cache affects updates |
| `test_nested_compiled` | Nested torch.compile submodules |
| `test_compiled_function` | @torch.compile on plain function |
| `test_compiled_function_decorator_with_args` | @torch.compile(...) with kwargs |
| `test_compiled_function_static_shape_variants` | dynamic=False creates multiple variants |
| `test_proxy_attribute_forwarding` | _CompiledModelProxy forwards attributes |
| `test_status` | status() function |
| `test_install_from_file` | install_from_file() API |
| `test_inference_only` | capture_backward=False |
| `test_larger_batch` | Shape handling with batch=32 |
| `test_nanochat` | Nanochat GPT forward+backward (CUDA, FA3) |
| `test_nanochat_training_loop` | Nanochat + MuonAdamW optimizer, 5 training steps (CUDA, FA3) |
| `test_auto_optimizer_capture` | optimizer.step() auto-captured |
| `test_auto_optimizer_capture_disabled` | capture_optimizer=False works |
| `test_auto_optimizer_training_matches_eager` | Auto-captured optimizer matches eager |
| `test_auto_optimizer_capture_uses_distinct_cache_keys_for_distinct_layouts` | Same-class optimizers with different layouts get separate caches |
| `test_registered_optimizer_preserves_first_step_return_value` | Registered custom optimizer preserves first step return value |
| `test_registered_optimizer_preserves_custom_step_after_capture` | Registered custom optimizer keeps using custom step_fn after capture |
| `test_registered_optimizer_forwards_step_args_on_first_capture` | Registered custom optimizer preserves call-time step args on first capture |
| `test_dynamic_batch_forward` | Forward with varying batch sizes |
| `test_dynamic_batch_backward` | Forward+backward with varying batch sizes |
| `test_dynamic_cache_reload` | Cache reload with dynamic graphs |
| `test_dynamic_training_loop` | Training with varying batch sizes |
| `test_optimizer_replay_sgd` | SGD optimizer replay matches eager over 5 steps |
| `test_optimizer_replay_adamw` | AdamW optimizer replay matches eager over 5 steps |
| `test_optimizer_replay_slot_info_stored_in_meta` | Replay metadata persisted in .meta file |
| `test_nanochat_optimizer_detected_as_inner_compiled` | MuonAdamW detected as using inner @torch.compile fns |
| `test_nanochat_optimizer_replay_inner_fns` | MuonAdamW inner fn capture+replay matches eager over 5 steps |
| `test_inner_fn_full_replay` | Full inner fn replay plan replaces optimizer loop on step 2+ |

### tests/test_install.py (11 tests)

Low-level `install()` function and capture pipeline.

| Test | What it verifies |
|------|-----------------|
| `test_capture_and_install_tracks_replaced_parameters` | Parameter tracking after install |
| `test_capture_and_install_tracks_replaced_submodule` | Submodule replacement tracking |
| `test_export_graph_to_python_uses_compact_source_comments` | Source comment formatting |
| `test_export_graph_to_python_uses_compact_module_headers` | Module header formatting |
| `test_shorten_source_path_dist_packages_does_not_use_local_common_root` | Path shortening logic |
| `test_load_aten_module_reloads_same_size_edit` | Edit-reload cycle for same-size edits |
| `test_extract_function_captures_standalone_fn` | extract_function on plain function |
| `test_extract_function_captures_nn_module` | extract_function on nn.Module |
| `test_extract_training_step_with_optimizer_capture` | extract_training_step + optimizer |
| `test_capture_optimizer_aten_uses_stable_slot_names` | Default optimizer slot names are `group{N}.param{M}` |
| `test_extract_training_step_with_custom_step_fn` | extract_training_step with custom step_fn |

### tests/test_install_cli.py (1 test)

| Test | What it verifies |
|------|-----------------|
| `test_nanochat_install_cli_smoke` | Full nanochat capture via CLI subprocess (`python -m torch_graph install`). Requires nanochat repo + tokenizer/data cache (see setup below). |

### tests/test_multi_step.py (11 tests)

Multi-step training and bit-identical gradient verification. All require CUDA.

| Test | What it verifies |
|------|-----------------|
| `test_multi_step_sgd` | 5 steps SGD, bit-identical losses |
| `test_multi_step_adamw` | 5 steps AdamW, near-identical losses (5e-5 tolerance) |
| `test_multi_step_batchnorm` | Multi-step with BatchNorm buffer mutations |
| `test_capture_io_recording` | record_real_tensors saves inputs/outputs/intermediates |
| `test_export_and_reimport` | Export to .py, import, contains forward+backward |
| `test_optimizer_capture` | capture_optimizer_aten on AdamW |
| `test_offload_saved_correctness` | offload_saved mode produces bit-identical grads (GPU only) |
| `test_cache_reload_multi_step` | Load from cache, train 3 steps |
| `test_replay_multi_step_sgd` | 5 steps SGD with optimizer replay matches eager |
| `test_replay_multi_step_adamw` | 5 steps AdamW with optimizer replay matches eager |
| `test_replay_multi_step_batchnorm` | Multi-step BatchNorm + optimizer replay matches eager |

### tests/test_inductor_comparison.py (5 tests)

Verify captured aten matches torch.compile(inductor) output. All require CUDA.

| Test | What it verifies |
|------|-----------------|
| `test_inductor_vs_aten_mlp` | MLP forward+backward: aten matches inductor |
| `test_inductor_vs_aten_conv` | ConvNet forward+backward: aten matches inductor |
| `test_inductor_vs_aten_batchnorm` | BatchNorm model: aten matches inductor |
| `test_inductor_vs_aten_multi_step` | 3-step training with optimizer replay: aten matches inductor |
| `test_inductor_vs_aten_nanogpt` | NanoGPT forward+backward: aten matches inductor |

### tests/test_triton_capture.py (3 tests)

Triton kernel capture and export. Require CUDA + Triton.

| Test | What it verifies |
|------|-----------------|
| `test_triton_kernel_capture_forward_only` | Triton ops appear in forward graph |
| `test_triton_kernel_export_roundtrip` | Export with kernels, verify @triton.jit decorator |
| `test_triton_auto_install_roundtrip` | Full auto_install pipeline with Triton kernels |

### tests/test_h5_roundtrip.py (5 tests)

H5 tensor storage dtype preservation. CPU-only.

| Test | What it verifies |
|------|-----------------|
| `test_fp16_all_values_roundtrip` | All 65536 fp16 bit patterns survive f32 roundtrip |
| `test_bf16_all_values_roundtrip` | All bf16 bit patterns survive f32 roundtrip |
| `test_fp16_h5_roundtrip` | FP16 tensors survive H5 write/read cycle |
| `test_bf16_h5_roundtrip` | BF16 tensors survive H5 write/read cycle |
| `test_blosc_compression_effective_for_bf16` | Blosc2 compression ratio >1.8x for bf16 |

### tests/test_ir_json.py (7 tests)

IR JSON generation and code export.

| Test | What it verifies |
|------|-----------------|
| `test_ir_json_python_matches_export_graph_to_python` | IR JSON code == export code |
| `test_ir_json_preserves_structured_args_and_kwargs` | Structured args/kwargs preserved |
| `test_ir_json_preserves_multi_output_returns` | Multi-output returns preserved |
| `test_ir_json_preserves_get_attr_tensor_literals` | get_attr tensor literals captured |
| `test_capture_to_ir_json_keeps_forward_backward_and_optimizer_sections` | All 3 sections in bundle |
| `test_ir_json_uses_original_source_line_when_source_map_is_available` | Source mapping works |
| `test_ir_json_renders_memory_format_kwargs` | Memory format kwargs rendered correctly |

### tests/test_visualizer_formats.py (4 tests)

Graph visualization (HTML + JSON only).

| Test | What it verifies |
|------|-----------------|
| `test_visualizer_supports_only_html_json` | Only to_html() and to_json() exist |
| `test_visualizer_combined_json_adds_backward_users` | Backward consumer links |
| `test_visualizer_combined_json_adds_backward_grad_targets` | Backward gradient targets |
| `test_visualizer_combined_json_adds_optimizer_links` | Optimizer param/grad/state links |

### tests/test_kbox_gen_h5.py (13 tests)

H5 dump, kbox generation, and replay script validation.

**CPU tests:**
| Test | What it verifies |
|------|-----------------|
| `test_kbox_group_scripts_cpu` | Group replay scripts produce bit-identical results |
| `test_forward_backward_full_chain_cpu` | Chain forward+backward group scripts |
| `test_inputs_only_recompute_cpu` | Recompute full forward from inputs_only H5 |
| `test_batchnorm_forward_cpu` | BatchNorm forward via H5 is bit-identical |
| `test_optimizer_capture` | Optimizer graph recording |
| `test_backward_section_script_cpu` | Section-level backward scripts |
| `test_backward_group_scripts_cpu` | Group-level backward scripts |
| `test_forward_section_script_cpu` | Section-level forward scripts |
| `test_section_script_relative_path` | Relative path handling in scripts |

**GPU tests (skip if no CUDA):**
| Test | What it verifies |
|------|-----------------|
| `test_kbox_group_scripts_gpu` | Group scripts on GPU |
| `test_gpu_tensor_storage_bit_identical` | GPU tensors bit-identical after H5 roundtrip |
| `test_backward_fx_graph_gpu` | Backward FX graph execution on GPU |
| `test_batchnorm_forward_gpu` | BatchNorm forward on GPU |

### tests/test_autoresearch_e2e.py (6 tests)

End-to-end autoresearch training verification. Requires nanochat repo + CUDA.

| Test | What it verifies |
|------|-----------------|
| `test_autoresearch_eager_baseline` | Eager training produces decreasing losses |
| `test_autoresearch_aten_vs_eager` | auto_install aten losses match eager over 5 steps (MuonAdamW with adamw+muon groups) |
| `test_autoresearch_verify_mode` | --verify mode records losses and produces training_summary.json |
| `test_autoresearch_muon_groups_captured` | Both adamw_step_fused and muon_step_fused are individually captured to disk |
| `test_autoresearch_full_inner_replay` | Full inner fn replay plan produces same losses as live replay over multiple steps |
| `test_autoresearch_params_are_live` | Sabotaged params produce matching aten/eager loss (proves params are live) |

### tests/test_nanochat_e2e.py (3 tests)

End-to-end nanochat training verification. Requires nanochat repo + CUDA + FA3.

| Test | What it verifies |
|------|-----------------|
| `test_nanochat_aten_vs_eager` | auto_install aten losses match eager over multiple steps |
| `test_nanochat_inner_fns_captured` | Inner compiled functions (adamw_step_fused, muon_step_fused) are captured to disk |
| `test_nanochat_live_params_verified` | Params are live (not frozen) — optimizer updates flow through correctly |

### tests/test_standalone.py (5 tests)

Standalone training loop generation. Verifies generated scripts produce correct losses. Requires CUDA; inner fn tests require nanochat repo.

| Test | What it verifies |
|------|-----------------|
| `test_standalone_generation` | Standalone script is generated from captured aten graphs (MLP + AdamW) |
| `test_standalone_matches_replay` | Standalone subprocess produces bit-identical losses vs live replay (MLP + AdamW, 4 steps) |
| `test_standalone_dynamic_shapes` | Standalone works with dynamic shape captures (SymInt resolution) |
| `test_standalone_inner_fn_generation` | Inner fn standalone script generated for MuonAdamW optimizer |
| `test_standalone_inner_fn_matches_replay` | Inner fn standalone produces same losses as live replay (per-step param/grad comparison) |

### tests/test_workarounds.py (3 tests)

CUDA workaround coverage for auto-applied PyTorch patches.

| Test | What it verifies |
|------|-----------------|
| `test_bitwise_or_workaround_preserves_bool_semantics` | Tensor bool OR fallback preserves bool behavior |
| `test_bitwise_or_workaround_preserves_bool_scalar_semantics` | Scalar bool OR fallback preserves bool behavior |
| `test_bitwise_or_workaround_matches_unsigned_integer_or` | uint16/32/64 OR fallback matches eager integer semantics |

---

## 2. run_tests.py (tensor verification suite)

Run with: `.venv/bin/python run_tests.py` (or `run_tests.py --quick` to skip NanoGPT)

**Current status: ALL TESTS PASSED**

Five verification sections:

1. **Determinism verification** — Run graph 2x with same inputs, compare all tensors. Models: Linear, MLP, Conv+BN+ReLU, LayerNorm+GELU, MultiheadAttn, NanoGPT.
2. **End-to-end verification** — Graph output vs real model output for all models above.
3. **Dynamic shapes** — MLP with dynamic=True, varying batch sizes.
4. **Op dump verification** — group_by=line, group_by=module, multi-section grouping, hide_views, H5 structure, replay scripts, PT format, dump_model_ops API, NanoGPT aliases, scripts_dir standalone files.
5. **Summary table** — FW Det / BW Det / E2E FW per model.

With `--quick`: skips NanoGPT model (faster, fewer GPU ops).

---

## 3. test_models.py (80 model recipes)

Run with: `.venv/bin/python test_models.py` (~3 min on GPU)

Captures aten graphs for 80+ model architectures and verifies the exported `.py` files are syntactically valid and executable. Uses a `VerifyPool` of pre-warmed Python subprocesses.

**Model categories:**
- **Wave 1 (30):** Basic architectures — MLP, ConvNet, LSTM, GRU, Transformer, ResNet, Autoencoder, MoE, UNet, RoPE, etc.
- **Wave 2 (30+):** Complex architectures — DenseNet, Inception, Ghost modules, Neural ODE, Spectral/Instance/Group norms, Linformer, Performer, Reformer, etc.
- **Torchvision (8+):** VGG-11, ViT-B/32, RegNet, MNASNet, Wide ResNet-50, DenseNet-121, AlexNet, ShuffleNet, SqueezeNet, Inception v3
- **HuggingFace (4):** BERT, DistilBERT, RoBERTa, GPT-2 (multi-fragment, graph breaks)
- **Function recipes (6):** Plain functions, partial model forward, custom attention, LSTM standalone, loss computation

Each recipe: `extract_training_step()` → export `.py` → verify syntax in subprocess → report pass/fail.

**Current status: 80/80 passed** (189.5s on GPU, as of 2026-03-11)

Filter specific models: `.venv/bin/python test_models.py --only resnet18,hf_gpt2`

---

## 4. test_fixes.py (legacy integration tests)

Run with: `.venv/bin/python test_fixes.py`

**Current status: 52/52 passed** (as of 2026-03-11)

Code-inspection and functional tests verifying specific historical fixes.

| Test | What it checks |
|------|---------------|
| 1. pyproject.toml build-backend | setuptools.build_meta is importable |
| 2. install.py kwargs ordering (2-arg) | Positional and keyword arg reordering |
| 3. install.py kwargs ordering (3-arg) | 3-arg positional/kwargs scrambling |
| 4. auto_install.py kwargs reorder logic | Named params, *args fallback, extra kwargs |
| 5. CUDA RNG state preservation | RNG save/restore around capture in `_capture_variant` |
| 6. keep_debug_dir in triton.py | Conditional cleanup uses shutil.rmtree |
| 7. tensor_dump multi-fragment code | No stale `[0]` indexing on graph lists |
| 8. tensor_dump functional (single fragment) | Returns results with kind='forward' |
| 9. tensor_dump multi-fragment functional | Graph breaks → >1 forward graphs |
| 10. dump_model_tensors multi-fragment | Multi-fragment forward keys |
| 11. Scan for stale [0] indexing | No `forward_graphs[0]` / `backward_graphs[0]` |
| 12. Signature capture ordering | `_orig_params` before forward replacement in install.py |
| 13. Backward with reversed kwargs | x.grad and y.grad match reference |
| 14. install via install.py kwargs | Capture → export → load → install with kwargs |
| 15. Code consistency | install.py owns helpers, auto_install delegates |

---

## 5. test.py (example/demo script)

Run with: `.venv/bin/python test.py`

**Current status: PASS**

Minimal demo script. Captures MLP and NanoGPT, generates HTML visualizations:
- `outputs/mlp_forward_v4.html`
- `outputs/nanogpt_forward_v4.html`

---

## 6. capture scripts (not tests, but verify real-world capture)

| Command | What it does | Time |
|---------|-------------|------|
| `make capture-nanochat` | Full nanochat capture: aten + HTML + H5 + kbox | ~30s (GPU) |
| `make capture-nanochat-cpu` | Nanochat capture on CPU (no FA3) | ~30s |
| `.venv/bin/python scripts/capture_modded_nanogpt.py` | Capture modded-nanogpt (543M params, FP8, Triton TMA) | ~60s (GPU) |

---

## Nanochat cache setup

`test_nanochat_install_cli_smoke` requires the nanochat tokenizer and data cache in `~/.cache/nanochat/`. To populate it (one-time, ~30s):

```bash
# From the repo root, with .venv active:
cd outputs/repos/nanochat

# Download 1 training shard + validation shard (~200MB)
PYTHONPATH=".:$PYTHONPATH" .venv/bin/python -m nanochat.dataset -n 1

# Train tokenizer on downloaded data (~0.2s)
PYTHONPATH=".:$PYTHONPATH" .venv/bin/python -m scripts.tok_train --max-chars 1000000
```

This creates `~/.cache/nanochat/tokenizer/` and `~/.cache/nanochat/base_data_climbmix/`.

---

## Coverage gaps (prioritized)

### Priority 1: Multi-fragment export

`_export_multi_fragment()` in `export.py` handles models with graph breaks (2+ forward/backward fragments). This is exercised only by `hf_gpt2` in test_models.py (which passes), but there is **no isolated test** that verifies:

- Fragment ordering: backward_0 corresponds to forward_(n-1) (reversed)
- The device fixup regex that adds `.to(_device)` to inline tensor constants — CLAUDE.md explicitly warns this regex must NOT match aten ops like `torch.arange` in source comments that span newlines. A greedy `[^)]*` match would produce broken code like `device=_device.to(_device)`.
- Per-fragment input/output chaining between fragments
- A simple synthetic 2-fragment model (rather than depending on hf_gpt2's specific graph break pattern)

**Risk:** If someone changes the device fixup regex without a targeted test, the breakage would only show up in hf_gpt2 — and only if that specific model happens to exercise the edge case.

### Priority 2: Backward saved tensor reordering

`_AtenGraph.backward()` in `install.py` reorders saved tensors from the interleaved order returned by forward into [non-tensor values (SymInts) first, then tensor values]. This is tested **implicitly** by every backward test that passes, but:

- No test has a model with mixed SymInt + tensor saved values in the backward graph
- No test explicitly verifies the reordering logic (they just check gradient correctness)
- The dynamic batch tests do use SymInts, providing some coverage

**Risk:** A model saving e.g. 3 SymInts + 5 tensors for backward could get wrong reordering. This would produce wrong gradients, not a crash.

### Priority 3: Custom ops tracking (`custom_ops.py`)

`custom_ops.py` intercepts custom op registrations (via `torch.library.Library()`, `torch.ops.load_library()`, and `ExtensionFileLoader`) so that exported aten files can reproduce them. It is **actively used by export.py** (`find_custom_op_namespaces`, `emit_custom_op_imports`) but has **zero test coverage**.

The modded-nanogpt capture script exercises this path in practice (it imports `torch_graph.custom_ops` explicitly for op tracking), but there's no pytest for it.

**Risk:** A model with custom C++ CUDA extensions would produce an exported `.py` file that's missing the imports needed to re-register those ops. The file would fail at load time, so it's not a silent correctness issue — but it's a broken user experience.

### Priority 4: Buffer mutation count

`_compute_num_mutations()` in `auto_install.py` determines how many buffer mutations (e.g. BatchNorm running_mean/running_var) are in the forward output. CLAUDE.md warns: "The num_mutations count must be correct or BatchNorm models silently produce wrong results."

Tested indirectly by `test_batchnorm` (2 mutations) and nanochat tests, but:

- No test for 0 mutations with real outputs (simplest case)
- No test for 3+ mutations
- No test that intentionally gets the count wrong and verifies it breaks

**Risk:** A new model architecture with unusual buffer mutation patterns could produce silently wrong results.

### Lower priority gaps

**FX-level graph APIs (`capture.py`, `inspector.py`, `editor.py`):** These are the older FX-level APIs predating the aten pipeline. They are **not used by any module in the aten pipeline** (not by export.py, auto_install.py, or install.py). They are only used by:
- `examples/` scripts (01_mnist_simple.py, 02_nanogpt.py, etc.)
- `visualizer.py` imports `CapturedGraph` as a type annotation
- Each other (inspector.py and editor.py import from capture.py)

They are exported in `__init__.py` but have zero test coverage. These should either be tested or removed from `__all__` to avoid presenting an untested public API.

**`extract_subgraph()` and `list_ops()`** in `export.py` — interactive utilities for browsing and extracting subgraphs. Exported in `__init__.py`, no tests. Would throw obvious errors if broken (not a silent correctness risk).

**`_workarounds.py`** — Registers CUDA kernels for unsigned-int bitwise ops (uint16/32/64 lshift, rshift, bitwise_or) that PyTorch <=2.10 lacks. No test. Only affects models using uint bitwise ops on CUDA.

**`tensor_dump.py` public API** — `compare_tensors()`, `verify_against_model()`, `trace_all_intermediates()` are exported in `__init__.py` but not directly tested. `run_tests.py` tests `dump_and_compare` and `dump_model_tensors` but not these specific functions.

**Error handling paths in `auto_install.py`** — ~17 `raise` statements for various error conditions (corrupt cache, missing forward function, bad config keys, etc.) are untested. These would produce crashes, not silent wrong results.
