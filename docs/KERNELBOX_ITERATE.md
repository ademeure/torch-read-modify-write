# KernelBox Iterate Workflow

This is the working path for running a real `torch-read-modify-write` capture
through `kernelbox-dev` iterate mode.

## What works now

- `torch_graph.kbox_gen` emits real `kbox iterate` test files:
  - each generated script defines `init_once()`
  - `init_once()` returns `inputs`, `expected`, and `watch_files`
  - editing the generated `run()` body works with `kbox iterate`
- `kernelbox-dev` can now:
  - load `.pt` / `.h5` data onto CPU when CUDA is unavailable
  - benchmark pure-PyTorch iterate files with wall-clock timing when CUDA is unavailable

## End-to-end example

Validated locally with a real `nanochat` capture stored at:

```text
outputs/nanochat_iterate_cpu/nanochat.h5
```

Generated iterate files:

```text
outputs/nanochat_iterate_cpu/kbox_iterate/forward_line_chain.py
outputs/nanochat_iterate_cpu/kbox_iterate/backward_line_chain.py
```

## 1. Capture a real H5 dump

Example: small `nanochat` model on CPU.

```bash
python3 - <<'PY'
import os, sys
sys.path.insert(0, '.')
sys.path.insert(0, 'recipes')

import torch
from nanochat_wrapper import _init, _build_tokenizer, _build_model, _make_token_pool
from torch_graph.export import capture_aten_graphs
from torch_graph.op_dump import dump_grouped_tensors

out_dir = 'outputs/nanochat_iterate_cpu'
os.makedirs(out_dir, exist_ok=True)

_init()
tokenizer = _build_tokenizer()
model, _config = _build_model(tokenizer.get_vocab_size(), depth=2, seq_len=16)
model.train()

token_pool = _make_token_pool(tokenizer, min_tokens=256)
torch.manual_seed(1337)
rows = [token_pool[:17]]
batch = torch.tensor(rows, dtype=torch.long)
x = batch[:, :-1].contiguous()
targets = batch[:, 1:].contiguous()

_, capture = capture_aten_graphs(
    model,
    x,
    targets=targets,
    run_backward=True,
    record_real_tensors=True,
)

dump_grouped_tensors(
    capture,
    os.path.join(out_dir, 'nanochat.h5'),
    group_by=['line'],
    which='both',
    include_params=True,
    replay_scripts=True,
    scripts_dir=os.path.join(out_dir, 'replay_scripts'),
)
PY
```

## 2. Generate iterate-compatible forward/backward scripts

Use chained section scripts first. They are much more useful than per-group
files when you want full forward or full backward correctness.

```bash
python3 -m torch_graph kbox \
  outputs/nanochat_iterate_cpu/nanochat.h5 \
  --chain \
  --section both \
  --strategy line \
  --out-dir outputs/nanochat_iterate_cpu/kbox_iterate
```

## 3. Run under kernelbox-dev iterate mode

Forward:

```bash
PYTHONPATH=/mnt/sharefs/user08/kernelbox-dev/python \
python3 /mnt/sharefs/user08/kernelbox-dev/tools/kbox_iterate.py \
  outputs/nanochat_iterate_cpu/kbox_iterate/forward_line_chain.py \
  --once --bench --warmup 1 --iters 3
```

Backward:

```bash
PYTHONPATH=/mnt/sharefs/user08/kernelbox-dev/python \
python3 /mnt/sharefs/user08/kernelbox-dev/tools/kbox_iterate.py \
  outputs/nanochat_iterate_cpu/kbox_iterate/backward_line_chain.py \
  --once --bench --warmup 1 --iters 3
```

`kbox iterate` validates automatically because the generated script returns
`expected` tensors from `init_once()`.

## 4. Iterate on edits

For live iteration, drop `--once`:

```bash
PYTHONPATH=/mnt/sharefs/user08/kernelbox-dev/python \
python3 /mnt/sharefs/user08/kernelbox-dev/tools/kbox_iterate.py \
  outputs/nanochat_iterate_cpu/kbox_iterate/forward_line_chain.py \
  --bench
```

Then edit the generated `run()` body in:

```text
outputs/nanochat_iterate_cpu/kbox_iterate/forward_line_chain.py
```

or:

```text
outputs/nanochat_iterate_cpu/kbox_iterate/backward_line_chain.py
```

The generated script also watches the backing H5 file, so regenerating the
tensor dump triggers a rerun.

## Notes

- `--strategy line` is the best first target for debugging and replay fidelity.
- `--strategy module` is available if you want larger grouped regions.
- Per-group scripts are still useful for narrow edits, but some groups do not
  have enough stored inputs to run in isolation.
- On a CUDA machine, the same flow uses GPU tensors and CUDA-event timing.
- On a CPU-only machine, file-backed iterate runs stay on CPU and benchmarking
  falls back to wall-clock timing.
