"""Reference CUDA kernels for every aten op.

One file per op. Each file has:
  KERNEL_SRC: raw CUDA kernel string
  WRAPPER_SRC: C++ wrapper for load_inline
  test(): compile, run with random tensors, compare to PyTorch
"""
