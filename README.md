# pytorch-auto-graph

Extract, visualize, edit, and **install** PyTorch computation graphs at the aten op level.

Given any PyTorch model or training script, this tool:

1. **Captures** the full forward + backward computation graph via TorchDynamo/aot_autograd
2. **Exports** it as a standalone Python file using only `torch.ops.aten.*` calls
3. **Installs** the captured graph back into the model — replacing `torch.compile` with editable aten ops
4. **Visualizes** the graph (HTML, JSON)
5. **Maps** aten ops to Triton kernels from `torch.compile`
6. **Dumps** all intermediate tensors and validates the exported graph against reference outputs
7. **Edits** graphs programmatically (swap ops, insert nodes, fuse patterns)

## Setup

Requires [uv](https://docs.astral.sh/uv/) and a CUDA GPU (tested on H100 with CUDA 13.x).

```bash
# One command: creates venv, installs all deps (PyTorch cu130, nanochat, etc.)
make setup

# Or step by step:
uv venv --python 3.12 .venv
uv pip install -e ".[all]"
```

### Nanochat dependencies

The nanochat model (karpathy/nanochat) requires additional packages for its tokenizer and data pipeline. `make setup` handles this automatically, or install manually:

```bash
uv pip install datasets fastapi ipykernel kernels matplotlib psutil \
  python-dotenv regex rustbpe scipy tabulate tiktoken tokenizers \
  transformers uvicorn wandb zstandard pyarrow
```

## Quick Start

```bash
# Run all tests (130+ pytest tests + tensor verification)
make test

# Capture nanochat with all outputs (aten, HTML, H5, kbox)
make capture-nanochat

# Everything at once
make all
```

## Auto Install (drop-in torch.compile replacement)

The primary workflow. Monkey-patches `torch.compile` so every compiled model automatically gets its aten graph captured to disk as an editable `.py` file. The model then runs using the aten graph instead of Inductor.

```bash
# Run any training script — torch.compile is transparently replaced:
.venv/bin/python -m torch_graph install train.py

# With a module:
.venv/bin/python -m torch_graph install -m scripts.base_train --depth=4 --num-iterations=20

# Options:
.venv/bin/python -m torch_graph install train.py --dynamic          # Dynamic shapes
.venv/bin/python -m torch_graph install train.py --graph            # HTML visualization
.venv/bin/python -m torch_graph install train.py --h5               # H5 tensor dump
.venv/bin/python -m torch_graph install train.py --recapture        # Force re-capture
```

### What happens:

1. `torch.compile(model)` is intercepted
2. On first call, forward+backward aten graphs are captured and saved to `.torch_graph_cache/`
3. The aten graph is installed as the model's forward — training runs using pure aten ops
4. **Edit the `.py` file** → changes take effect on next run (no recompile needed)
5. Train and eval modes get separate aten files for independent editing

**Capture is bit-identical**: the captured aten graph produces the exact same forward output and gradients as eager execution, including with Flash Attention 3 on Hopper GPUs.

### Nanochat capture

Generates forward/backward aten graphs, interactive HTML visualizations, H5 tensor dumps with replay scripts, and kernelbox test scripts:

```bash
make capture-nanochat
# Or directly:
.venv/bin/python scripts/capture_nanochat.py --depth=4 --seq-len=64
```

Output in `outputs/nanochat/`:
- `nanochat_aten.py` — Editable aten forward+backward (2000+ lines)
- `nanochat_forward.html` — Interactive HTML visualization
- `nanochat_backward.html` — Backward graph visualization
- `nanochat_forward.json` — JSON graph export
- `nanochat.h5` — H5 tensor dump (forward+backward, all intermediates)
- `nanochat_kbox/` — Kernelbox test scripts
- `nanochat_scripts/` — Standalone replay scripts per module/line

## Programmatic API

```python
from torch_graph import capture_aten_graphs, export_aten_program
import torch

model = torch.nn.Linear(8, 4)
x = torch.randn(2, 8)

# Capture + export
output, capture = capture_aten_graphs(model, x)
export_aten_program(capture, "outputs/linear_aten.py")
```

```python
from torch_graph.export import capture_aten_graphs
from torch_graph.install import install

model = MyModel()
out, capture = capture_aten_graphs(model, sample_input)
install(model, capture)  # model.forward now uses aten ops
```

## Tests

```bash
make test          # All tests (pytest + tensor verification + NanoGPT)
make test-quick    # Fast tests only (skip NanoGPT, skip nanochat)
make test-models   # 80 model recipe tests (~3 min GPU)
```

Or run directly:
```bash
.venv/bin/python -m pytest tests/ -v           # 130+ pytest tests
.venv/bin/python run_tests.py                   # Tensor verification suite
.venv/bin/python test_models.py                 # 80 model recipes
.venv/bin/python test_models.py --only resnet18,hf_gpt2  # Specific models
```

## Architecture

```
torch_graph/
├── auto_install.py  # Drop-in torch.compile replacement (main workflow)
├── install.py       # Install aten graphs back into models via autograd.Function
├── _install_cli.py  # CLI: python -m torch_graph install
├── export.py        # Aten-level export with forward+backward, source annotations
├── capture.py       # FX graph capture via custom TorchDynamo backend
├── auto.py          # Zero-modification extraction from arbitrary scripts
├── triton.py        # Triton kernel capture + aten-to-kernel mapping
├── tensor_dump.py   # Tensor dumping and output verification
├── op_dump.py       # Grouped tensor dumps to H5 with replay scripts
├── kbox_gen.py      # Generate kernelbox test scripts from H5 dumps
├── inspector.py     # Graph analysis (op counts, shapes, dependencies)
├── editor.py        # Programmatic graph editing with undo
├── visualizer.py    # Graph visualization (HTML, JSON)
├── _utils.py        # Shared utilities (RecordingInterpreter, is_fake, etc.)
├── __init__.py      # Public API
└── __main__.py      # CLI entry point
```

## How It Works

1. **FX Trace** — `torch.compile` with a custom backend captures source locations, module paths, and stack traces
2. **AOT Export** — `aot_autograd` decomposes into aten ops, producing separate forward and backward graphs with primal-to-parameter mapping
3. **Code Export** — Standalone `.py` file with aten ops, source annotations, and optional test harness
4. **Install** — `torch.autograd.Function` wraps forward/backward; parameters flow live (not frozen) so optimizer updates work

### Key concepts

- **Primals**: Inputs to the aten graph (`primals_1`, `primals_2`, ...) mapped to parameter names like `self.fc1.weight`
- **Multi-variant dispatch**: Each call pattern (train/eval, different arg counts) gets its own aten file. Eval variants run eagerly unless the file has been user-modified.
- **Dynamic shapes**: With `dynamic=True`, varying dimensions (e.g. batch size) use symbolic values (`s77`) so one graph works for any size

### Dynamic Shapes

Dynamic shapes are supported via `dynamic=True` (the default in `auto_install`). Symbolic dimensions allow a single captured graph to handle varying input sizes (batch size, sequence length, etc.) without re-capture:

```python
from torch_graph.export import capture_aten_graphs
output, capture = capture_aten_graphs(model, x, dynamic=True)
# The graph now works for any batch size / sequence length
```

For the CLI:
```bash
.venv/bin/python -m torch_graph install train.py --dynamic
```

### Explain API

One-liner to capture, inspect, and optionally verify a model:

```python
from torch_graph import explain
result = explain(model, x, verify=True)
# result.op_counts, result.op_categories, result.shapes, result.summary()
```

### Other workflows

```bash
# Direct extraction (no install):
.venv/bin/python -m torch_graph train.py --output-dir outputs/
.venv/bin/python -m torch_graph model.py --class NanoGPT -o outputs/
```

```python
# Verification:
from torch_graph.tensor_dump import dump_and_compare, verify_against_model
dump_and_compare(model, x, run_backward=True)       # Run 2x, compare intermediates
verify_against_model(model, x, run_backward=True)   # Compare aten vs eager
```

## Requirements

- Python >= 3.10 (tested with 3.12)
- PyTorch >= 2.10 (cu130)
- CUDA GPU (H100/Hopper for Flash Attention 3; other GPUs work without FA3)
- [uv](https://docs.astral.sh/uv/) for dependency management
