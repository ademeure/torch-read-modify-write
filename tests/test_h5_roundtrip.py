"""Test H5 tensor storage roundtrip for all dtypes."""

import os
import sys
import tempfile

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from torch_graph.op_dump import _torch_to_numpy


def test_fp16_all_values_roundtrip():
    """All 65536 fp16 bit patterns must survive f32→numpy→f32→fp16 losslessly.

    This covers normals, denormals, zeros, infs.  NaN payloads are allowed
    to change as long as NaN-ness is preserved.
    """
    all_bits = torch.arange(0, 65536, dtype=torch.int16)
    all_fp16 = all_bits.view(torch.float16)

    np_data, stored_as = _torch_to_numpy(all_fp16.cpu())
    assert stored_as == "float32_from_float16"
    assert np_data.dtype == np.float32

    # Roundtrip back
    back_fp16 = torch.from_numpy(np_data).half()
    orig_bits = all_fp16.view(torch.int16)
    back_bits = back_fp16.view(torch.int16)

    # Non-NaN must be bitwise identical
    is_nan = all_fp16.isnan()
    non_nan = ~is_nan
    assert (orig_bits[non_nan] == back_bits[non_nan]).all(), \
        "Some non-NaN fp16 values changed during f32 roundtrip"

    # NaN must stay NaN
    assert back_fp16[is_nan].isnan().all(), "Some NaN values lost NaN-ness"


def test_bf16_all_values_roundtrip():
    """All 65536 bf16 bit patterns must survive f32→numpy→f32→bf16 losslessly."""
    all_bits = torch.arange(0, 65536, dtype=torch.int16)
    all_bf16 = all_bits.view(torch.bfloat16)

    np_data, stored_as = _torch_to_numpy(all_bf16.cpu())
    assert stored_as == "float32_from_bfloat16"
    assert np_data.dtype == np.float32

    back_bf16 = torch.from_numpy(np_data).bfloat16()
    orig_bits = all_bf16.view(torch.int16)
    back_bits = back_bf16.view(torch.int16)

    is_nan = all_bf16.isnan()
    non_nan = ~is_nan
    assert (orig_bits[non_nan] == back_bits[non_nan]).all(), \
        "Some non-NaN bf16 values changed during f32 roundtrip"
    assert back_bf16[is_nan].isnan().all(), "Some NaN values lost NaN-ness"


def test_fp16_h5_roundtrip():
    """FP16 tensors survive the full H5 write→read cycle."""
    h5py = pytest.importorskip("h5py")

    t = torch.randn(100, 100, dtype=torch.float16)
    np_data, stored_as = _torch_to_numpy(t.cpu())

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        path = f.name
    try:
        with h5py.File(path, "w") as f:
            ds = f.create_dataset("t", data=np_data)
            ds.attrs["torch_dtype"] = "torch.float16"
            ds.attrs["stored_as"] = stored_as

        with h5py.File(path, "r") as f:
            loaded = torch.from_numpy(f["t"][()]).half()

        assert torch.equal(t, loaded), "FP16 H5 roundtrip mismatch"
    finally:
        os.unlink(path)


def test_bf16_h5_roundtrip():
    """BF16 tensors survive the full H5 write→read cycle."""
    h5py = pytest.importorskip("h5py")

    t = torch.randn(100, 100, dtype=torch.bfloat16)
    np_data, stored_as = _torch_to_numpy(t.cpu())

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        path = f.name
    try:
        with h5py.File(path, "w") as f:
            ds = f.create_dataset("t", data=np_data)
            ds.attrs["torch_dtype"] = "torch.bfloat16"
            ds.attrs["stored_as"] = stored_as

        with h5py.File(path, "r") as f:
            loaded = torch.from_numpy(f["t"][()]).bfloat16()

        assert torch.equal(t, loaded), "BF16 H5 roundtrip mismatch"
    finally:
        os.unlink(path)


def test_blosc_compression_effective_for_bf16():
    """BF16 stored as f32 with blosc should be much smaller than raw f32."""
    h5py = pytest.importorskip("h5py")
    hdf5plugin = pytest.importorskip("hdf5plugin")

    t = torch.randn(1000, 1000, dtype=torch.bfloat16)
    np_data, stored_as = _torch_to_numpy(t.cpu())
    comp = hdf5plugin.Blosc2(cname="lz4", clevel=5,
                             filters=hdf5plugin.Blosc2.SHUFFLE)

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        path = f.name
    try:
        with h5py.File(path, "w") as f:
            f.create_dataset("compressed", data=np_data,
                             compression=comp, chunks=np_data.shape)
            f.create_dataset("raw", data=np_data)

        size = os.path.getsize(path)
        raw_size = np_data.nbytes  # 4MB for 1M float32

        with h5py.File(path, "r") as f:
            comp_id = f["compressed"].id
            raw_id = f["raw"].id
            comp_bytes = comp_id.get_storage_size()
            raw_bytes = raw_id.get_storage_size()

        ratio = raw_bytes / comp_bytes
        # bf16→f32 zeroes bottom 16 bits → expect ~2x compression minimum
        assert ratio > 1.8, f"Compression ratio {ratio:.2f}x too low for bf16→f32"
    finally:
        os.unlink(path)
