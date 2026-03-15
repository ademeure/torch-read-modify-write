#!/usr/bin/env python3
"""Test auto_install with a simple model + a @torch.compile'd function."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Patch torch.compile BEFORE any model code ──
import torch_graph.auto_install
torch_graph.auto_install.configure(
    cache_dir="/tmp/torch_graph_test_cache",
    verbose=True,
    num_real_outputs=1,
)

import torch
import torch.nn as nn

# ═══════════════════════════════════════════════════════════════════════
# Simple model
# ═══════════════════════════════════════════════════════════════════════

class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


# ═══════════════════════════════════════════════════════════════════════
# Test 1: Model compilation
# ═══════════════════════════════════════════════════════════════════════

print("=" * 60)
print("TEST 1: Model torch.compile → auto_install")
print("=" * 60)

model = TinyMLP()
x = torch.randn(2, 8)

# Reference output (eager)
ref_out = model(x)
print(f"Reference output: {ref_out.shape}, sum={ref_out.sum().item():.6f}")

# Reset model to same weights
model2 = TinyMLP()
model2.load_state_dict(model.state_dict())

# This calls our patched torch.compile
model2 = torch.compile(model2)
compiled_out = model2(x)
print(f"Compiled output:  {compiled_out.shape}, sum={compiled_out.sum().item():.6f}")

# Verify
diff = (ref_out - compiled_out).abs().max().item()
print(f"Max diff: {diff:.2e}")
assert diff < 1e-5, f"Output mismatch: {diff}"

# Test backward
loss = compiled_out.sum()
loss.backward()
print(f"Backward OK, fc1.weight.grad shape: {model2.fc1.weight.grad.shape}")

print("\n✓ Test 1 passed: model install works")


# ═══════════════════════════════════════════════════════════════════════
# Test 2: Second call uses cache
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("TEST 2: Second instantiation loads from cache")
print("=" * 60)

model3 = TinyMLP()
model3.load_state_dict(model.state_dict())
model3 = torch.compile(model3)
cached_out = model3(x)
diff2 = (ref_out - cached_out).abs().max().item()
print(f"Max diff from cache: {diff2:.2e}")
assert diff2 < 1e-5, f"Cache mismatch: {diff2}"
print("✓ Test 2 passed: cache loading works")


# ═══════════════════════════════════════════════════════════════════════
# Test 3: @torch.compile on a function
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("TEST 3: @torch.compile on standalone function")
print("=" * 60)

@torch.compile
def my_optimizer_step(param, grad, lr):
    return param - lr * grad

p = torch.randn(4, requires_grad=False)
g = torch.randn(4)
result = my_optimizer_step(p, g, 0.01)
expected = p - 0.01 * g
diff3 = (result - expected).abs().max().item()
print(f"Max diff: {diff3:.2e}")
assert diff3 < 1e-6, f"Function mismatch: {diff3}"
print("✓ Test 3 passed: function replacement works")


# ═══════════════════════════════════════════════════════════════════════
# Test 4: Nested model (submodules compiled independently)
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("TEST 4: Nested model with compiled submodules")
print("=" * 60)

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 8)

    def forward(self, x):
        return torch.relu(self.fc(x))


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        return self.fc(x)


class Nested(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.compile(Encoder())
        self.decoder = torch.compile(Decoder())

    def forward(self, x):
        h = self.encoder(x)
        return self.decoder(h)


nested = Nested()
x4 = torch.randn(2, 8)
out4 = nested(x4)
print(f"Nested output: {out4.shape}, sum={out4.sum().item():.6f}")

loss4 = out4.sum()
loss4.backward()
print(f"Nested backward OK, encoder.fc.weight.grad: {nested.encoder.fc.weight.grad is not None}")
print(f"                     decoder.fc.weight.grad: {nested.decoder.fc.weight.grad is not None}")
print("✓ Test 4 passed: nested compiled submodules work")


# ═══════════════════════════════════════════════════════════════════════
# Test 5: User-modified aten file loaded from disk
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("TEST 5: User-modified aten file (simulated custom kernel)")
print("=" * 60)

# The TinyMLP cache file already exists. Let's modify it to add a scaling
# factor — simulating a user who edited the aten file to add custom logic.
import time
cache_dir = "/tmp/torch_graph_test_cache"
cache_file = os.path.join(cache_dir, "TinyMLP_44a11fe65886_aten.py")

# Read the cached file
with open(cache_file) as f:
    aten_src = f.read()

# Modify: scale the output by 2.0 (simulates a custom kernel edit)
aten_src = aten_src.replace(
    "return (addmm_1,",
    "# User modification: scale output by 2x\n"
    "    scaled = aten.mul(addmm_1, 2.0)\n"
    "    return (scaled,",
)

with open(cache_file, 'w') as f:
    f.write(aten_src)

# Touch the file to make it newer than the .meta (triggers user-modified detection)
time.sleep(1)
os.utime(cache_file, None)

# Now create a fresh model and compile it — should load our modified file
model5 = TinyMLP()
model5.load_state_dict(model.state_dict())
model5 = torch.compile(model5)
modified_out = model5(x)

# The output should be 2x the reference
expected_2x = ref_out * 2.0
diff5 = (modified_out - expected_2x).abs().max().item()
print(f"Modified output sum:  {modified_out.sum().item():.6f}")
print(f"Expected (2x ref):    {expected_2x.sum().item():.6f}")
print(f"Max diff: {diff5:.2e}")
assert diff5 < 1e-4, f"User-modified file mismatch: {diff5}"
print("✓ Test 5 passed: user-modified aten file loaded correctly")


# ═══════════════════════════════════════════════════════════════════════
# Test 6: Explicit install_from_file API
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("TEST 6: Explicit install_from_file API")
print("=" * 60)

# Restore the unmodified cached file for Encoder
encoder_cache = os.path.join(cache_dir, "Encoder_5d5c6aac48fb_aten.py")
model6 = Encoder()
model6.load_state_dict(nested.encoder.state_dict())
torch_graph.auto_install.install_from_file(model6, encoder_cache)
out6 = model6(x4)
print(f"install_from_file output: {out6.shape}")
assert out6.shape == (2, 8)
print("✓ Test 6 passed: explicit install_from_file works")


# ═══════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("STATUS")
print("=" * 60)
print(torch_graph.auto_install.status())

print("\n✓ All tests passed!")
