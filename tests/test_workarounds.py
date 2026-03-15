import torch

from torch_graph._workarounds import _bitwise_or_kernel


def test_bitwise_or_workaround_preserves_bool_semantics():
    lhs = torch.tensor([True, False, True, False], dtype=torch.bool)
    rhs = torch.tensor([False, False, True, True], dtype=torch.bool)

    result = _bitwise_or_kernel(lhs, rhs)
    expected = torch.bitwise_or(lhs, rhs)

    assert result.dtype == torch.bool
    assert torch.equal(result, expected)


def test_bitwise_or_workaround_preserves_bool_scalar_semantics():
    lhs = torch.tensor([True, False, False], dtype=torch.bool)

    result = _bitwise_or_kernel(lhs, True)
    expected = torch.bitwise_or(lhs, torch.tensor(True, dtype=torch.bool))

    assert result.dtype == torch.bool
    assert torch.equal(result, expected)


def test_bitwise_or_workaround_matches_unsigned_integer_or():
    lhs = torch.tensor([0, 1, 255, 2**15, 2**63], dtype=torch.uint64)
    rhs = torch.tensor([1, 2, 3, 2**15 - 1, 7], dtype=torch.uint64)

    result = _bitwise_or_kernel(lhs, rhs)
    expected = torch.bitwise_or(lhs, rhs)

    assert result.dtype == torch.uint64
    assert torch.equal(result, expected)
