"""Universal input fuzzer for kbox test files.

Given baseline inputs (from init_once) and a reference function,
generates variant inputs and computes expected outputs automatically.

Each seed produces a mix of random values and special float values
injected at random positions. Seed 0 guarantees every special value
appears at least once per float tensor (dense injection). Higher seeds
vary the injection density from 5-25%.

Always runs both modes:
  - Full specials (including NaN/inf) — tests propagation behavior
  - Safe specials (no NaN/inf) — tests actual computation

Usage::

    from kernelbox.fuzz import fuzz_inputs

    baseline = [torch.randn(1024, device="cuda")]
    for label, variant in fuzz_inputs(baseline, seeds=10):
        expected = reference(variant)
        actual = run(variant, kernel)
        verify(actual, expected)
"""
from __future__ import annotations

import math

import torch

# Special float values that trigger edge cases in GPU kernels.
SPECIAL_VALUES = [
    0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 4.0, -4.0,
    100.0, -100.0, 1e-7, -1e-7, 1e7, -1e7,
    1e-45, -1e-45,          # subnormals (float32)
    1.18e-38, -1.18e-38,    # near min normal (float32)
    3.4e38, -3.4e38,        # near max (float32)
    float("nan"), float("inf"), float("-inf"),
]

# Same but without NaN/inf — tests actual computation, not propagation.
SAFE_SPECIAL_VALUES = [v for v in SPECIAL_VALUES if not (math.isnan(v) or math.isinf(v))]


def fuzz_inputs(baseline_inputs, seeds=10):
    """Yield (label, variant_inputs) from baseline inputs.

    For each seed, yields TWO variants: one with full specials (NaN/inf
    included) and one safe (no NaN/inf). Total variants = 2 * seeds.

    Non-float tensors (int indices, bool masks) are cloned unchanged.

    Seed 0 guarantees every special value appears at least once per
    tensor. Higher seeds use random injection at varying density.

    Args:
        baseline_inputs: list of tensors from init_once()
        seeds: number of seed values (yields 2x this many variants)

    Yields:
        (str, list[Tensor]) — (description, variant inputs)
    """
    for seed in range(seeds):
        yield _gen_mixed(baseline_inputs, seed, SPECIAL_VALUES, "fuzz")
        yield _gen_mixed(baseline_inputs, seed, SAFE_SPECIAL_VALUES, "safe")


def _gen_mixed(baseline, seed, specials, prefix):
    """One variant: seeded randn base + special value injection.

    seed=0: dense injection — every special value appears at least once
            per tensor (cyclic placement), guaranteeing exhaustive coverage.
    seed>0: random injection at varying density (5-25%).
    """
    device = baseline[0].device if baseline else "cuda"
    g = torch.Generator(device=device).manual_seed(seed)

    specials_t = torch.tensor(specials, dtype=torch.float32, device=device)
    n_specials = len(specials_t)

    variant = []
    for t in baseline:
        if not t.is_floating_point():
            variant.append(t.clone())
            continue

        s = specials_t.to(t.dtype)

        # Random base (randn)
        v = torch.randn(t.shape, dtype=t.dtype, device=t.device, generator=g)

        if seed == 0:
            # Dense: cycle all specials through the tensor, guaranteeing
            # every value appears at least once (if numel >= n_specials).
            flat = v.flatten()
            n = flat.numel()
            indices = torch.arange(n, device=t.device)
            flat[indices % n_specials < n_specials] = s[indices[:n] % n_specials]
            # Keep ~50% as randn for mixed behavior
            keep_mask = torch.rand(n, device=t.device, generator=g) < 0.5
            randn_vals = torch.randn(n, dtype=t.dtype, device=t.device, generator=g)
            flat[keep_mask] = randn_vals[keep_mask]
            # But ensure at least one of each special survives
            for i in range(min(n_specials, n)):
                flat[i] = s[i]
            v = flat.view(t.shape)
        else:
            # Random injection at varying density
            frac = 0.05 + 0.20 * ((seed * 7 + 3) % 11) / 10
            mask = torch.rand(t.shape, device=t.device, generator=g) < frac
            if mask.any():
                idx = torch.randint(n_specials, t.shape, device=t.device,
                                    generator=g)
                v[mask] = s[idx[mask]]

        variant.append(v)

    return (f"{prefix}_{seed}", variant)
