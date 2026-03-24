"""Universal input fuzzer for kbox test files.

Three modes per seed, yielded in order:
  fuzz_N  — randn + all specials (NaN/inf/subnormals/boundary) injected
  safe_N  — randn + safe specials only (no NaN/inf)
  rand_N  — pure randn, no specials at all

Total variants = 3 * seeds.

Cross-product guarantee region covers all combinations of special values
across float inputs (e.g. 33^2=1089 for binary ops). All float inputs
share the same guarantee region so every pair is tested. Seed rotates
values so different seeds cover different cross-product slices.

    from kernelbox.fuzz import fuzz_inputs
    for label, variant in fuzz_inputs(baseline_inputs, seeds=10):
        expected = reference(variant)
        actual = run(variant, kernel)
"""
from __future__ import annotations

import math
import torch

# Non-NaN special float values (float32).
_FINITE_SPECIALS = [
    0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 4.0, -4.0,
    100.0, -100.0, 1e-7, -1e-7, 1e7, -1e7,
    1.401298464324817e-45,               # smallest subnormal (0x00000001)
    -1.401298464324817e-45,
    1.1754942106924411e-38,              # largest subnormal (0x007FFFFF)
    -1.1754942106924411e-38,
    1.1754943508222875e-38,              # smallest normal (0x00800000)
    -1.1754943508222875e-38,
    1e8, -1e8,              # large (won't overflow in short dot products)
    float("inf"), float("-inf"),
]

# NaN variants with different bit patterns (payload, sign).
# Built via numpy uint32→int32 view to preserve exact bits, because
# Python float can't represent different NaN payloads and torch.tensor()
# rejects unsigned ints > 2^31-1 for dtype=int32.
_NAN_BITS = [
    0x7FC00000,  # +qNaN (default, payload=0)
    0xFFC00000,  # -qNaN (sign bit set)
    0x7FC00001,  # +qNaN payload=1
    0x7FFFFFFF,  # +qNaN all payload bits set
    0x7F800001,  # +sNaN (quiet bit=0, payload=1) — GPU won't trap but bit pattern differs
]

SAFE_SPECIAL_VALUES = [v for v in _FINITE_SPECIALS
                       if not math.isinf(v)]


def _build_specials(device):
    """Build full specials tensor with exact NaN bit patterns."""
    import numpy as np
    finite = torch.tensor(_FINITE_SPECIALS, dtype=torch.float32, device=device)
    nan_np = np.array(_NAN_BITS, dtype=np.uint32).view(np.int32)
    nan_bits = torch.from_numpy(nan_np).to(device)
    nans = nan_bits.view(torch.float32)
    return torch.cat([finite, nans])


def fuzz_inputs(baseline_inputs, seeds=10):
    """Yield (label, variant_inputs) pairs. Total = 3 * seeds.

    Each seed yields three variants: full specials (fuzz_N), safe specials
    without NaN/inf (safe_N), and pure randn (rand_N).

    Non-float tensors (int indices, bool masks) are cloned unchanged.
    Only float32 specials are injected — bf16/fp16 inputs get the same
    specials cast down, which may lose subnormal/boundary precision.

    Requires list/tuple inputs (dict inputs are not supported).
    """
    if not baseline_inputs:
        return
    if not isinstance(baseline_inputs, (list, tuple)):
        return  # dict inputs not supported
    device = baseline_inputs[0].device
    all_specials = _build_specials(device)
    safe_specials = torch.tensor(SAFE_SPECIAL_VALUES, dtype=torch.float32,
                                 device=device)
    for seed in range(seeds):
        yield _gen(baseline_inputs, seed, all_specials, "fuzz")
        yield _gen(baseline_inputs, seed, safe_specials, "safe")
        yield _gen_rand(baseline_inputs, seed)


def _gen(baseline, seed, specials_f32, prefix):
    device = baseline[0].device
    g = torch.Generator(device=device).manual_seed(seed)
    ns = len(specials_f32)
    frac = 0.05 + 0.20 * ((seed * 7 + 3) % 11) / 10

    # Count float inputs to size the cross-product guarantee region.
    # All float inputs share the same region length so every combination
    # of specials across inputs is covered.
    n_float = sum(1 for t in baseline if t.is_floating_point())
    full_xprod = ns ** n_float

    variant = []
    float_idx = 0
    for t in baseline:
        if not t.is_floating_point():
            variant.append(t.clone())
            continue

        # Note: .to(t.dtype) may lose subnormal/boundary precision for bf16/fp16.
        s = specials_f32.to(t.dtype)
        flat = torch.randn(t.numel(), dtype=t.dtype, device=t.device, generator=g)
        n = flat.numel()

        # Inject specials at random positions
        mask = torch.rand(flat.shape, device=device, generator=g) < frac
        idx = torch.randint(ns, flat.shape, device=device, generator=g)
        flat[mask] = s[idx[mask]]

        # Cross-product guarantee: all float inputs use the same region
        # length (ns^n_float). Input j uses stride ns^j within that region.
        # Seed rotates values so different seeds cover different slices
        # (critical when full_xprod > n and one seed can't cover all).
        guarantee_len = min(full_xprod, n)
        stride = ns ** float_idx
        offset = (seed * guarantee_len) % n if guarantee_len < n else 0
        positions = torch.arange(guarantee_len, device=device)
        values = s[((positions // stride) + seed) % ns]
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
    device = baseline[0].device
    g = torch.Generator(device=device).manual_seed(seed + 1_000_000)

    variant = []
    for t in baseline:
        if not t.is_floating_point():
            variant.append(t.clone())
            continue
        variant.append(torch.randn(t.shape, dtype=t.dtype, device=t.device,
                                   generator=g))

    return (f"rand_{seed}", variant)
