"""Universal input fuzzer for kbox test files.

Three modes per seed, yielded in order:
  fuzz_N  — randn + all specials (NaN/inf/subnormals/boundary) injected
  safe_N  — randn + safe specials only (no NaN/inf)
  rand_N  — pure randn, no specials at all

Cross-product guarantee region covers all combinations of special values
across float inputs (e.g. 27^2=729 for binary ops). Offset by seed so
different seeds test the cross-product against different random neighbors.

    from kernelbox.fuzz import fuzz_inputs
    for label, variant in fuzz_inputs(baseline_inputs, seeds=10):
        expected = reference(variant)
        actual = run(variant, kernel)
"""
from __future__ import annotations

import math
import torch

SPECIAL_VALUES = [
    0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 4.0, -4.0,
    100.0, -100.0, 1e-7, -1e-7, 1e7, -1e7,
    1e-45, -1e-45,          # subnormals (float32)
    1.18e-38, -1.18e-38,    # near min normal (float32)
    3.4e38, -3.4e38,        # near max (float32)
    float("nan"), float("nan") * -1,  # +NaN, -NaN (sign bit differs)
    float("inf"), float("-inf"),
]

SAFE_SPECIAL_VALUES = [v for v in SPECIAL_VALUES
                       if not (math.isnan(v) or math.isinf(v))]


def fuzz_inputs(baseline_inputs, seeds=10):
    """Yield (label, variant_inputs) pairs. Total = 3 * seeds.

    Each seed yields three variants: full specials, safe specials, pure
    random. Non-float tensors are cloned unchanged.
    """
    for seed in range(seeds):
        yield _gen(baseline_inputs, seed, SPECIAL_VALUES, "fuzz")
        yield _gen(baseline_inputs, seed, SAFE_SPECIAL_VALUES, "safe")
        yield _gen_rand(baseline_inputs, seed)


def _gen(baseline, seed, specials, prefix):
    device = baseline[0].device if baseline else "cuda"
    g = torch.Generator(device=device).manual_seed(seed)
    s_all = torch.tensor(specials, dtype=torch.float32, device=device)
    ns = len(s_all)
    frac = 0.05 + 0.20 * ((seed * 7 + 3) % 11) / 10

    variant = []
    float_idx = 0
    for t in baseline:
        if not t.is_floating_point():
            variant.append(t.clone())
            continue

        s = s_all.to(t.dtype)
        flat = torch.randn(t.numel(), dtype=t.dtype, device=t.device, generator=g)
        n = flat.numel()

        # Inject specials at random positions
        mask = torch.rand(flat.shape, device=device, generator=g) < frac
        idx = torch.randint(ns, flat.shape, device=device, generator=g)
        flat[mask] = s[idx[mask]]

        # Cross-product guarantee: input j uses stride ns^j, covering all
        # combinations across inputs. Offset by seed so different seeds
        # place the guarantee against different random neighborhoods.
        stride = ns ** float_idx
        guarantee_len = min(ns * stride, n)
        offset = (seed * guarantee_len) % n
        positions = torch.arange(guarantee_len, device=device)
        values = s[(positions // stride) % ns]
        # Write with wraparound
        end = offset + guarantee_len
        if end <= n:
            flat[offset:end] = values
        else:
            split = n - offset
            flat[offset:] = values[:split]
            flat[:end - n] = values[split:]

        float_idx += 1
        variant.append(flat.view(t.shape))

    return (f"{prefix}_{seed}", variant)


def _gen_rand(baseline, seed):
    """Pure randn, no specials. Tests normal-range computation."""
    device = baseline[0].device if baseline else "cuda"
    # Use a distinct seed space so rand variants differ from fuzz/safe
    g = torch.Generator(device=device).manual_seed(seed + 1_000_000)

    variant = []
    for t in baseline:
        if not t.is_floating_point():
            variant.append(t.clone())
            continue
        variant.append(torch.randn(t.shape, dtype=t.dtype, device=t.device,
                                   generator=g))

    return (f"rand_{seed}", variant)
