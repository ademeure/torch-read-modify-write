"""Reference CUDA kernels for ~150 aten ops — the complete PyTorch op vocabulary.

One file per op, using the kernelbox init_once/run pattern:
  KERNEL_SRC: raw CUDA kernel (extern "C" __global__ only, no C++ wrappers)
  init_once(): inputs + expected via torch.ops.aten.*
  run(inputs, kernel): execute via kernelbox — kernel(*inputs) or custom params

Run one:  kbox iterate torch_graph/cuda_ref_kernels/aten_add.py --once
Run all:  python torch_graph/cuda_ref_kernels/run_all_tests.py

Categories covered:
  Elementwise unary (49):  abs, neg, exp, log, sqrt, relu, gelu, silu, sigmoid, tanh, ...
  Elementwise binary (12): add, sub, mul, div, pow, maximum, minimum, ...
  Comparison (6):          eq, ne, gt, ge, lt, le
  Ternary/conditional (6): where, clamp, lerp, addcmul, addcdiv, masked_fill
  Backward gradient (5):   threshold_backward, gelu_backward, silu_backward, ...
  Reductions (8):          sum, mean, amax, amin, prod, var, argmax, argmin
  Scan (2):                cumsum, cumprod
  Sort/search (2):         sort, topk
  Matmul/linalg (8):       mm, bmm, addmm, dot, mv, outer, baddbmm, linear
  Normalization (5):       _softmax, _log_softmax, native_layer_norm, native_batch_norm, native_group_norm
  Layout/view (14):        transpose, t, permute, view, reshape, clone, contiguous, ...
  Tensor creation (6):     arange, zeros, ones, full, eye, linspace
  Indexing (4):             gather, scatter, index_select, index_add
  Concat/split (3):        cat, stack, split
  Rearrange (5):           flip, roll, repeat, tril, triu
  Convolution (1):         convolution (naive conv2d)
  Pooling (3):             max_pool2d, avg_pool2d, adaptive_avg_pool2d
  Embedding (1):           embedding
  Loss (2):                nll_loss_forward, mse_loss
  Attention (1):           scaled_dot_product_attention (naive reference)
  Dropout (1):             native_dropout
  Padding (1):             constant_pad_nd
  Type/identity (4):       _to_copy, fill, alias, detach

All tests compare against torch.ops.aten.* (the actual aten ops), not high-level PyTorch APIs.
"""
