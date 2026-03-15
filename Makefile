# pytorch-auto-graph Makefile
# Single-command setup, testing, and nanochat capture

# Use .venv Python if available, otherwise system Python
PYTHON = $(shell test -x .venv/bin/python && echo .venv/bin/python || echo python3)
UV = $(shell command -v uv 2>/dev/null)

# Triton needs libcuda.so; add CUDA stubs if the unversioned symlink is missing
export LIBRARY_PATH := $(shell \
	if ! test -e /usr/lib/x86_64-linux-gnu/libcuda.so 2>/dev/null; then \
		for d in /usr/local/cuda/targets/x86_64-linux/lib/stubs \
		         /usr/local/cuda-13.1/targets/x86_64-linux/lib/stubs \
		         /usr/local/cuda-13.0/targets/x86_64-linux/lib/stubs; do \
			test -e "$$d/libcuda.so" && echo "$$d:$(LIBRARY_PATH)" && break; \
		done; \
	else echo "$(LIBRARY_PATH)"; fi)

.PHONY: setup test test-quick test-models capture-nanochat all clean help

all: test capture-nanochat ## Run all tests + capture nanochat (GPU)

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Setup ─────────────────────────────────────────────────────────────

setup: .setup_stamp ## Install everything (deps + nanochat repo)

.setup_stamp: pyproject.toml
ifdef UV
	$(UV) venv --python 3.12 .venv 2>/dev/null || true
	$(UV) pip install -e ".[all]"
else
	@echo "uv not found, using pip with system Python"
	pip install torch==2.10.0+cu130 --index-url https://download.pytorch.org/whl/cu130
	pip install -e ".[all]"
endif
	@# Clone nanochat if not present
	@if [ ! -d outputs/repos/nanochat/nanochat ]; then \
		echo "Cloning nanochat..."; \
		mkdir -p outputs/repos; \
		git clone --depth 1 https://github.com/karpathy/nanochat.git outputs/repos/nanochat; \
	fi
ifdef UV
	$(UV) pip install datasets fastapi ipykernel kernels matplotlib psutil \
		python-dotenv regex rustbpe scipy tabulate tiktoken tokenizers \
		transformers uvicorn wandb zstandard pyarrow
else
	pip install datasets fastapi ipykernel kernels matplotlib psutil \
		python-dotenv regex rustbpe scipy tabulate tiktoken tokenizers \
		transformers uvicorn wandb zstandard pyarrow
endif
	@touch .setup_stamp

# ── Testing ───────────────────────────────────────────────────────────

test: setup ## Run all tests (pytest + run_tests.py)
	$(PYTHON) -m pytest tests/ -v --tb=short
	$(PYTHON) run_tests.py

test-quick: setup ## Run fast tests only (pytest + run_tests.py --quick)
	$(PYTHON) -m pytest tests/ -v --tb=short -k "not nanochat"
	$(PYTHON) run_tests.py --quick

test-models: setup ## Run model recipe tests (~3 min GPU)
	$(PYTHON) test_models.py

# ── Nanochat capture ──────────────────────────────────────────────────

capture-nanochat: setup ## Capture nanochat: aten graphs + HTML + H5 + kbox
	$(PYTHON) scripts/capture_nanochat.py --device=cuda

capture-nanochat-cpu: setup ## Capture nanochat on CPU (no flash attention)
	$(PYTHON) scripts/capture_nanochat.py --device=cpu

# ── Cleanup ───────────────────────────────────────────────────────────

clean: ## Remove caches and generated outputs
	rm -rf .torch_graph_cache __pycache__ outputs/nanochat .setup_stamp .venv/.installed
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

clean-all: clean ## Remove everything including venv and cloned repos
	rm -rf .venv outputs/repos
