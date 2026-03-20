"""Shared test input generation for aten reference kernels.

All make_inputs() functions use these helpers to generate special values,
seeded random inputs, and cross-product pairs for binary ops.
"""
import torch

# Special float values: zeros, normals, subnormals, large, nan, inf
SPECIAL_VALUES = [
    0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, 4.0,
    100.0, -100.0, 1e-7, 1e7,
    1e-45, -1e-45,           # subnormals (smallest positive fp32)
    1.18e-38, -1.18e-38,     # near smallest normal
    float("nan"), float("inf"), float("-inf"),
]


def special_1d(n, device="cuda"):
    """Generate a 1D tensor of special values, repeated to fill n elements."""
    v = torch.tensor(SPECIAL_VALUES, device=device)
    return v.repeat((n + len(v) - 1) // len(v))[:n]


def special_pair(n, device="cuda"):
    """Generate cross-product pairs of special values for binary ops.
    Returns (a, b) where every combination of special values appears."""
    v = torch.tensor(SPECIAL_VALUES, device=device)
    m = len(v)
    a = v.repeat_interleave(m).repeat((n + m * m - 1) // (m * m))[:n]
    b = v.repeat(m).repeat((n + m * m - 1) // (m * m))[:n]
    return a, b


def seeded_randn(n, seed, device="cuda"):
    """Seeded normal random tensor."""
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(n, device=device, generator=g)


def seeded_rand(n, seed, device="cuda"):
    """Seeded uniform [0,1) random tensor."""
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.rand(n, device=device, generator=g)


def make_1d(n=1024, seed=1, device="cuda"):
    """Standard 1-input generation: seed=0 → special, seed>0 → random."""
    if seed == 0:
        return special_1d(n, device)
    return seeded_randn(n, seed, device)


def make_pair(n=1024, seed=1, device="cuda"):
    """Standard 2-input generation: seed=0 → cross-product specials, seed>0 → random."""
    if seed == 0:
        return special_pair(n, device)
    g = torch.Generator(device=device).manual_seed(seed)
    a = torch.randn(n, device=device, generator=g)
    b = torch.randn(n, device=device, generator=g)
    return a, b
