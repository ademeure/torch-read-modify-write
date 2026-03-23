"""Universal input fuzzer for kbox test files.

Three modes per seed, yielded in order:
  fuzz_N  — randn + all specials (NaN/inf/subnormals/boundary) injected
  safe_N  — randn + safe specials only (no NaN/inf)
  rand_N  — pure randn, no specials at all

Cross-product guarantee region covers all combinations of special values
across float inputs (e.g. 30^2=900 for binary ops). Offset by seed so
different seeds test the cross-product against different random neighbors.

    from kernelbox.fuzz import fuzz_inputs
    for label, variant in fuzz_inputs(baseline_inputs, seeds=10):
        expected = reference(variant)
        actual = run(variant, kernel)
"""
from __future__ import annotations

import math
import torch

# Non-NaN special float values.
_FINITE_SPECIALS = [
    0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 4.0, -4.0,
    100.0, -100.0, 1e-7, -1e-7, 1e7, -1e7,
    1e-45, -1e-45,                      # smallest subnormal (0x00000001)
    1.1754942106924411e-38,              # largest subnormal (0x007FFFFF)
    -1.1754942106924411e-38,
    1.1754943508222875e-38,              # smallest normal (0x00800000)
    -1.1754943508222875e-38,
    3.4e38, -3.4e38,        # near max (float32)
    float("inf"), float("-inf"),
]

# NaN variants with different bit patterns (payload, sign).
# Built via int32→float32 view to preserve exact bits.
_NAN_BITS = [
    0x7FC00000,  # +qNaN (default, payload=0)
    0xFFC00000,  # -qNaN (sign bit set)
    0x7FFFFFFF,  # +qNaN all payload bits set
    0x7F800001,  # +sNaN (quiet bit=0, payload=1) — GPU won't trap but bit pattern differs
]

SAFE_SPECIAL_VALUES = [v for v in _FINITE_SPECIALS
                       if not math.isinf(v)]


def _build_specials(device):
    """Build full specials tensor with exact NaN bit patterns."""
    import numpy as np
    finite = torch.tensor(_FINITE_SPECIALS, dtype=torch.float32, device=device)
    # Use numpy uint32 → torch int32 view to avoid Python int overflow
    nan_np = np.array(_NAN_BITS, dtype=np.uint32).view(np.int32)
    nan_bits = torch.from_numpy(nan_np).to(device)
    nans = nan_bits.view(torch.float32)
    return torch.cat([finite, nans])


def fuzz_inputs(baseline_inputs, seeds=10):
    """Yield (label, variant_inputs) pairs. Total = 3 * seeds.

    Each seed yields three variants: full specials, safe specials, pure
    random. Non-float tensors are cloned unchanged.
    """
    device = baseline_inputs[0].device if baseline_inputs else "cuda"
    all_specials = _build_specials(device)
    safe_specials = torch.tensor(SAFE_SPECIAL_VALUES, dtype=torch.float32,
                                 device=device)
    for seed in range(seeds):
        yield _gen(baseline_inputs, seed, all_specials, "fuzz")
        yield _gen(baseline_inputs, seed, safe_specials, "safe")
        yield _gen_rand(baseline_inputs, seed)


def _gen(baseline, seed, specials_f32, prefix):
    device = baseline[0].device if baseline else "cuda"
    g = torch.Generator(device=device).manual_seed(seed)
    ns = len(specials_f32)
    frac = 0.05 + 0.20 * ((seed * 7 + 3) % 11) / 10

    variant = []
    float_idx = 0
    for t in baseline:
        if not t.is_floating_point():
            variant.append(t.clone())
            continue

        s = specials_f32.to(t.dtype)
        flat = torch.randn(t.numel(), dtype=t.dtype, device=t.device, generator=g)
        n = flat.numel()

        # Inject specials at random positions
        mask = torch.rand(flat.shape, device=device, generator=g) < frac
        idx = torch.randint(ns, flat.shape, device=device, generator=g)
        flat[mask] = s[idx[mask]]

        # Cross-product guarantee: input j uses stride ns^j, covering all
        # combinations across inputs. Seed rotates both position (offset)
        # and values (shift) so different seeds test different cross-product
        # slices against different random neighborhoods.
        stride = ns ** float_idx
        guarantee_len = min(ns * stride, n)
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
    device = baseline[0].device if baseline else "cuda"
    g = torch.Generator(device=device).manual_seed(seed + 1_000_000)

    variant = []
    for t in baseline:
        if not t.is_floating_point():
            variant.append(t.clone())
            continue
        variant.append(torch.randn(t.shape, dtype=t.dtype, device=t.device,
                                   generator=g))

    return (f"rand_{seed}", variant)
