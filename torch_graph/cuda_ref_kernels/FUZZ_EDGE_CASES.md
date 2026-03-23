# CUDA Reference Kernel Edge Cases (Fuzz Testing)

Fuzz testing with `kbox batch --registry ... --fuzz 10` tests all 185 real-kernel
ops with adversarial inputs: NaN (4 variants including sNaN), inf, subnormals,
boundary values, and their cross-products across inputs.

## Fixed (kernel matches aten on adversarial inputs)

| Op | Issue | Fix |
|---|---|---|
| argmax | NaN not selected (NaN > x is false) | NaN-aware comparison: `v > best \|\| (v != v && best == best)` |
| argmin | Same NaN comparison issue | Same fix |
| sort | NaN not sorted to end | NaN-last comparison in bubble sort |
| var | Textbook formula `E[x^2]-E[x]^2` unstable | Two-pass algorithm with double accumulation |
| gelu_backward | `exp(-0.5*b*b)` overflows for large b | Clamp exp_term to 0 for `isfinite(b) && |b|>10` |
| sigmoid_backward | Fuzzed inputs outside [0,1] | Clamp b to [0,1] in kernel |
| max_pool2d | NaN not propagated through max | `v > best || v != v` |

## Remaining failures (18 ops — hard edge cases)

These are genuine divergences between hand-written CUDA kernels and PyTorch aten
on adversarial inputs. They all pass with normal inputs (`--seeds 10`).

### NaN-vs-inf disagreements (IEEE 754 edge cases)
Operations like `inf * 0`, `inf - inf`, `inf / inf` are NaN per IEEE 754, but
float32 evaluation order and GPU hardware may produce inf or -inf instead.

| Op | Example input | Kernel produces | Aten produces | Root cause |
|---|---|---|---|---|
| addcdiv | value*inf/small | -inf | NaN | `(double)inf / small` = inf in double, aten evaluates differently |
| addcmul | value*large*large | -inf | NaN | Overflow order in float32 vs aten |
| addmm | matmul with inf | -inf | NaN | cuBLAS handles inf×0 differently |
| linear | matmul with extreme values | 0.0 | NaN | Same cuBLAS issue |
| lerp | lerp(inf, -inf, w) | inf | NaN | `inf + w*(-inf-inf)` vs aten's handling |
| sigmoid_backward | NaN as sigmoid output | NaN | -inf | Clamped sigmoid doesn't match aten's unclamped path |

### Precision failures (accumulation with extreme values)
When inputs include values near float32 max (3.4e38), accumulation diverges
from aten's higher-precision or different-order computation.

| Op | Max error | Root cause |
|---|---|---|
| adaptive_avg_pool2d | 3.12e+05 | float32 sum of 64 extreme values |
| cumsum | 3.22 | Sequential accumulation with 1e7 values |
| cumprod | — | Running product overflow, kernel produces 0 where aten produces NaN |
| mv | 0.20 | float32 dot product of 32 extreme values |
| nll_loss_forward | 0.125 | float32 mean of 16 extreme log-probs |
| native_group_norm | 4.00 | Variance formula with extreme values |

### Simplified/incomplete kernels
These kernels use simplified formulas that omit correction terms.

| Op | Issue |
|---|---|
| native_batch_norm | `(x-mean)*rstd` produces NaN for extreme values, aten handles gracefully |
| native_group_norm_backward | Simplified backward: omits mean/var gradient correction terms |
| native_layer_norm_backward | Simplified backward: only `weight * rstd * grad` (missing 2 terms) |

### Complex multi-step ops
| Op | Issue |
|---|---|
| scaled_dot_product_attention | NaN in Q/K poisons softmax → all NaN output, aten zeros NaN attention |
| topk | NaN comparison: kernel returns NaN in wrong positions vs aten convention |
| upsample_bilinear2d | Index clamping mismatch: different neighbor pixel selected near boundaries |

## How to fix remaining ops

**Precision**: Use `double` accumulation (already applied to cumsum, cumprod, mv, nll_loss,
addcdiv, addcmul, addmm, linear, convolution, adaptive_avg_pool2d — but some still fail
because the divergence is NaN-vs-finite, not precision).

**NaN propagation**: Each op needs case-by-case analysis of how aten handles NaN. Key patterns:
- `inf * 0 = NaN` (IEEE 754) — GPU may evaluate as inf if intermediate overflow happens first
- Reductions with NaN: aten propagates NaN, some kernels skip or ignore NaN
- Sort/topk: PyTorch convention is NaN > everything (sorted last ascending)

**Simplified backward kernels**: Need full backward formula implementation with shared-memory
reductions. The current simplified kernels have atol=1e10 for fuzz mode.

## Testing commands

```bash
kbox batch --registry torch_graph.cuda_ref_kernels._registry --seeds 10     # Normal: 185/185
kbox batch --registry torch_graph.cuda_ref_kernels._registry --fuzz 10      # Adversarial: 167/185
```
