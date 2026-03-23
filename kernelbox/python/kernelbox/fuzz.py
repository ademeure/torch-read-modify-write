"""Universal input fuzzer for kbox test files.

Generates variant inputs from baseline shapes: seeded randn with special
float values (NaN, inf, subnormals, zeros, boundary) injected at random
positions. Always tests both with and without NaN/inf.

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
    float("nan"), float("inf"), float("-inf"),
]

SAFE_SPECIAL_VALUES = [v for v in SPECIAL_VALUES
                       if not (math.isnan(v) or math.isinf(v))]


def fuzz_inputs(baseline_inputs, seeds=10):
    """Yield (label, variant_inputs) pairs. Total = 2 * seeds.

    Each seed yields two variants: full specials (with NaN/inf) and safe
    (without). Non-float tensors are cloned unchanged. Every special value
    is guaranteed to appear at least once per float tensor.
    """
    for seed in range(seeds):
        yield _gen(baseline_inputs, seed, SPECIAL_VALUES, "fuzz")
        yield _gen(baseline_inputs, seed, SAFE_SPECIAL_VALUES, "safe")


def _gen(baseline, seed, specials, prefix):
    device = baseline[0].device if baseline else "cuda"
    g = torch.Generator(device=device).manual_seed(seed)
    s_all = torch.tensor(specials, dtype=torch.float32, device=device)
    ns = len(s_all)
    frac = 0.05 + 0.20 * ((seed * 7 + 3) % 11) / 10

    variant = []
    for t in baseline:
        if not t.is_floating_point():
            variant.append(t.clone())
            continue

        s = s_all.to(t.dtype)
        flat = torch.randn(t.numel(), dtype=t.dtype, device=t.device, generator=g)

        # Inject specials at random positions
        mask = torch.rand(flat.shape, device=device, generator=g) < frac
        idx = torch.randint(ns, flat.shape, device=device, generator=g)
        flat[mask] = s[idx[mask]]

        # Guarantee every special appears at least once
        n = flat.numel()
        for i in range(min(ns, n)):
            flat[i] = s[i]

        variant.append(flat.view(t.shape))

    return (f"{prefix}_{seed}", variant)
