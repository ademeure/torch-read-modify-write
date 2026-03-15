# torch-graph Progress Log

## Phase 1: Bug Fixes (DONE)
- [x] Fix pytorch_wlm graph break (is_causal=True in recipe)
- [x] Fix export graph-break resilience (clear primal_names + real_inputs)
- [x] Fix dump CLI recipe support (detect setup() function)

## Phase 2: Expand Model Coverage (DONE)
- [x] Wave 4: 12 torchvision models (ResNet, MobileNet, EfficientNet, etc.)
- [x] Wave 5: 9 models (3 HuggingFace + SimpleViT + MoE + Perceiver + causal transformer + DenseNet + AlexNet)
- [x] AOT set_ fallback for HuggingFace/Swin dynamo graph capture
- [x] `_resolve_loss()` helper for HF model outputs
- [x] Total: 78 recipes, 78/78 passing

## Phase 3: Performance Optimization (DONE)
- [x] Benchmark at multiple scales:
  - 2-layer (491K): 1.5s
  - 4-layer (988K): 1.8s
  - 8-layer (2M): 3.4s
  - 12-layer/GPT-2 (286M): 6.2s
  - 20-layer (896M): 11.5s
  - 26-layer/1.68B: 14.4s capture, 14.7s export (29.1s total)
- [x] All captures under 20s target (capture only, excluding export)
- [x] Export generates compact .py (1.1MB for 1.68B) + .pt (6.3GB weights)

## Phase 4: nanochat End-to-End (DONE)
- [x] Clone karpathy/nanochat repo
- [x] Recipe file with MuonAdamW optimizer (Muon + AdamW parameter grouping)
- [x] Full forward+backward capture at all scales
- [x] **Numerical correctness**: 0 difference in loss, gradients, parameters
  after 3 training steps vs real model
- [x] Op dump: source-annotated groups showing RoPE, SDPA, relu², VE, etc.
- [x] All nanochat features captured: RoPE, QK norm, relu², value embeds,
  sliding window, logit softcapping, residual/x0 lambdas, GQA

## Phase 5: Advanced nanochat Features (DONE)
- [x] MuonAdamW optimizer integration with proper parameter grouping
- [x] Triton kernel capture via inductor (39 kernels: 12 Triton + 27 extern)
- [x] SFT loss capture (masked targets with ignore_index=-1)
  - **Numerical correctness**: 0 difference after 3 training steps
  - Op dump shows proper source annotations (RoPE, SDPA, cross_entropy)
- [x] RL / GRPO loss capture (per-token NLL × advantages policy gradient)
  - **Numerical correctness**: 0 difference after 3 training steps
  - Custom loss_fn correctly computes policy gradient objective
- [x] `--setup-fn` CLI flag for recipe variants (extract_repo.py + dump)
- [x] Total: 78 recipes, 78/78 passing

## Phase 6: HF Decoder Fix + Polish
- [x] Fix `_safe_deepcopy_gm`: temporarily disable FakeTensorMode dispatch
  modes during `copy.deepcopy(gm)` to avoid FakeTensor contamination
- [x] Fix `get_attr` node export: emit tensor constants instead of function calls
- [x] GPT2 now captures + exports end-to-end (2 FW + 2 BW graphs via graph breaks)
- [x] Viewer orphan node layout fix (push depth-0 nodes to consumers)
- [x] Total: 79 recipes, 79/79 passing

## Known Limitations
- **Swin Transformer**: aten.set_.source_Storage in shifted window attention
- **FP8**: Requires CUDA (torch._scaled_mm); can't test on CPU
- **1.5B+ export**: .pt files are 6+ GB, need separate disk
- **GPT2/T5 graph breaks**: DynamicCache causes 2 graph fragments (captured, not merged)

## Key Insights
- TransformerEncoder with causal masks needs is_causal=True to avoid graph break
- Graph breaks cause: (1) wrong primal mapping, (2) wrong real_inputs in export
- HF encoder models (BERT, DistilBERT, RoBERTa) work perfectly
- HF decoder models: `_disable_current_modes()` fixes FakeTensor deepcopy issue
- Muon optimizer: Polar Express orthogonalization is fully traceable
- nanochat capture is numerically exact (0 difference vs real model)
- Performance scales linearly with model depth (~0.5s per layer)
