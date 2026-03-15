"""PyTorch workarounds applied automatically on import.

Registers missing or broken CUDA kernels so that torch.compile and
aot_autograd work transparently with ops that PyTorch doesn't yet support.

All patches are idempotent, use ``torch.library.impl``, and work inside
``torch.compile``.  Add new workarounds as functions below and call them
from ``apply_all()`` at the bottom.
"""

import logging
import warnings

import torch

logger = logging.getLogger(__name__)

# ─── Workaround 1: unsigned-int bitwise ops (PyTorch ≤ 2.10) ────────────
#
# PyTorch lacks CUDA kernels for lshift, rshift, bitwise_or on uint16/32/64.
# We register kernels that view-cast unsigned→signed (zero-copy), call the
# signed-type kernel, and view-cast back.
#
# Recursion avoidance: torch.library.impl replaces the ENTIRE CUDA dispatch
# key (all dtypes), so signed types also route through our kernel.
# - For shifts: we implement aten::__lshift__ via torch.bitwise_left_shift
#   (a separate aten op with its own dispatch entry) — no recursion.
# - For bitwise_or: the CUDA dispatch key is replaced for every dtype, so the
#   fallback must preserve bool/signed semantics too. We implement OR via
#   De Morgan's law using bitwise_not/bitwise_and, which are separate ops and
#   therefore avoid recursion while remaining exact for bool and integers.

_UNSIGNED_TO_SIGNED = {
    torch.uint16: torch.int16,
    torch.uint32: torch.int32,
    torch.uint64: torch.int64,
}


def _make_shift_kernel(torch_fn):
    def kernel(self, other):
        signed = _UNSIGNED_TO_SIGNED.get(self.dtype)
        if signed is not None:
            if isinstance(other, torch.Tensor):
                o_signed = _UNSIGNED_TO_SIGNED.get(other.dtype)
                if o_signed is not None:
                    other = other.view(o_signed)
            return torch_fn(self.view(signed), other).view(self.dtype)
        return torch_fn(self, other)
    return kernel


def _bitwise_or_kernel(self, other):
    orig_dtype = self.dtype
    if not isinstance(other, torch.Tensor):
        other = torch.tensor(other, dtype=orig_dtype, device=self.device)
    signed = _UNSIGNED_TO_SIGNED.get(orig_dtype)
    if signed is not None:
        self = self.view(signed)
        other_signed = _UNSIGNED_TO_SIGNED.get(other.dtype)
        if other_signed is not None:
            other = other.view(other_signed)
        return torch.bitwise_not(
            torch.bitwise_and(
                torch.bitwise_not(self),
                torch.bitwise_not(other),
            )
        ).view(orig_dtype)
    return torch.bitwise_not(
        torch.bitwise_and(
            torch.bitwise_not(self),
            torch.bitwise_not(other),
        )
    )


def _patch_uint_bitwise_ops():
    registrations = [
        (_make_shift_kernel(torch.bitwise_left_shift), [
            "aten::__lshift__.Scalar", "aten::__lshift__.Tensor",
        ]),
        (_make_shift_kernel(torch.bitwise_right_shift), [
            "aten::__rshift__.Scalar", "aten::__rshift__.Tensor",
        ]),
        (_bitwise_or_kernel, [
            "aten::bitwise_or.Scalar", "aten::bitwise_or.Tensor",
        ]),
    ]
    registered = []
    for impl_fn, aten_names in registrations:
        for aten_name in aten_names:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    torch.library.impl(aten_name, "cuda")(impl_fn)
                registered.append(aten_name)
            except Exception:
                pass
    if registered:
        logger.info("Workaround: registered uint bitwise CUDA ops: %s",
                     ", ".join(registered))


# ─── Apply all workarounds ───────────────────────────────────────────────

_applied = False


def apply_all():
    """Apply all PyTorch workarounds. Idempotent."""
    global _applied
    if _applied:
        return
    _applied = True
    _patch_uint_bitwise_ops()


apply_all()
