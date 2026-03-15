"""Simple training loop for NanoGPT - the auto-extractor should capture this."""
import torch
from model import NanoGPT

# Create model and data
model = NanoGPT(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Quick training loop
for step in range(3):
    idx = torch.randint(0, 64, (2, 16))
    logits = model(idx)
    loss = logits.sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"Step {step}: loss={loss.item():.4f}")

print("Training done!")
