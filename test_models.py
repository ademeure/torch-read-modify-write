#!/usr/bin/env python3
"""Validate extract_repo on a wide variety of PyTorch models.

Each model is a self-contained recipe dict. We call extract_training_step()
directly (no subprocess) so errors surface immediately.
"""

import os
import sys
import math
import traceback
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from torch_graph.extract import extract_training_step, extract_function

import subprocess
import threading
import queue

# Force immediate progress logs even when stdout is a pipe.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(line_buffering=True, write_through=True)
    except Exception:
        pass

# Worker script: imports torch (the expensive part) at startup, then waits
# for a script path on stdin and runs it via runpy.
_VERIFY_WORKER_CODE = '''\
import sys, runpy, traceback
import torch
line = sys.stdin.readline().strip()
if not line:
    sys.exit(0)
sys.argv = line.split()
try:
    runpy.run_path(sys.argv[0], run_name="__main__")
except SystemExit as e:
    if e.code:
        sys.exit(e.code)
except Exception:
    traceback.print_exc()
    sys.exit(1)
'''


class VerifyPool:
    """Pre-warms Python subprocesses with torch already imported.

    Popen() returns immediately after fork; the child process spends ~1.5s
    importing torch in the background.  By the time we need to verify an
    aten file (after 2-3s of extraction work), a warm worker is already
    waiting on stdin.
    """

    def __init__(self, n_warm=2):
        self._workers = []
        for _ in range(n_warm):
            self._workers.append(self._make_worker())

    def _make_worker(self):
        return subprocess.Popen(
            [sys.executable, "-u", "-c", _VERIFY_WORKER_CODE],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True,
        )

    def verify(self, py_file, device="cpu", timeout=120):
        """Run py_file in a pre-warmed worker. Returns (success, error_msg)."""
        if not self._workers:
            self._workers.append(self._make_worker())
        worker = self._workers.pop(0)
        # Start warming a replacement immediately
        self._workers.append(self._make_worker())
        try:
            cmd = py_file if device != "cpu" else f"{py_file} --cpu"
            stdout, stderr = worker.communicate(
                input=f"{cmd}\n", timeout=timeout,
            )
            if worker.returncode != 0:
                return False, f"Verify failed (rc={worker.returncode}):\n{stderr[-2000:]}"
            return True, None
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.communicate()
            return False, "Verify timed out"

    def shutdown(self):
        for w in self._workers:
            try:
                w.stdin.close()
                w.kill()
                w.wait(timeout=2)
            except Exception:
                pass
        self._workers.clear()


# ═══════════════════════════════════════════════════════════════════════
# Model definitions
# ═══════════════════════════════════════════════════════════════════════


# 1. Simple MLP
def mlp_recipe():
    model = nn.Sequential(
        nn.Linear(64, 128), nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(),
        nn.Linear(128, 10),
    )
    x = torch.randn(4, 64)
    targets = torch.randint(0, 10, (4,))
    return {
        "model": model,
        "sample_args": (x,),
        "loss_fn": lambda out: F.cross_entropy(out, targets),
    }


# 2. Conv + BatchNorm + Pool (LeNet-like)
def convnet_recipe():
    class ConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(16)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(32)
            self.pool = nn.AdaptiveAvgPool2d(4)
            self.fc = nn.Linear(32 * 4 * 4, 10)

        def forward(self, x):
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    model = ConvNet()
    x = torch.randn(4, 1, 28, 28)
    targets = torch.randint(0, 10, (4,))
    return {
        "model": model,
        "sample_args": (x,),
        "loss_fn": lambda out: F.cross_entropy(out, targets),
    }


# 3. LSTM-based sequence model
def lstm_recipe():
    class LSTMModel(nn.Module):
        def __init__(self, vocab_size=100, embed_dim=32, hidden_dim=64, n_layers=2):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, embed_dim)
            self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers, batch_first=True, dropout=0.0)
            self.fc = nn.Linear(hidden_dim, vocab_size)

        def forward(self, x):
            e = self.embed(x)
            out, _ = self.lstm(e)
            return self.fc(out[:, -1, :])

    model = LSTMModel()
    x = torch.randint(0, 100, (4, 20))
    targets = torch.randint(0, 100, (4,))
    return {
        "model": model,
        "sample_args": (x,),
        "loss_fn": lambda out: F.cross_entropy(out, targets),
    }


# 4. GRU-based model
def gru_recipe():
    class GRUModel(nn.Module):
        def __init__(self, vocab_size=80, embed_dim=24, hidden_dim=48):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, embed_dim)
            self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, vocab_size)

        def forward(self, x):
            e = self.embed(x)
            out, _ = self.gru(e)
            return self.fc(out[:, -1, :])

    model = GRUModel()
    x = torch.randint(0, 80, (4, 16))
    targets = torch.randint(0, 80, (4,))
    return {
        "model": model,
        "sample_args": (x,),
        "loss_fn": lambda out: F.cross_entropy(out, targets),
    }


# 5. Transformer Encoder (text classification style)
def transformer_encoder_recipe():
    class TransformerClassifier(nn.Module):
        def __init__(self, vocab_size=200, d_model=64, nhead=4, n_layers=2, n_classes=5):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            self.pos = nn.Embedding(128, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=128,
                batch_first=True, dropout=0.0,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.fc = nn.Linear(d_model, n_classes)

        def forward(self, x):
            seq_len = x.size(1)
            pos_ids = torch.arange(seq_len, device=x.device)
            e = self.embed(x) + self.pos(pos_ids)
            h = self.encoder(e)
            return self.fc(h.mean(dim=1))

    model = TransformerClassifier()
    x = torch.randint(0, 200, (4, 32))
    targets = torch.randint(0, 5, (4,))
    return {
        "model": model,
        "sample_args": (x,),
        "loss_fn": lambda out: F.cross_entropy(out, targets),
    }


# 6. Residual MLP (skip connections)
def residual_mlp_recipe():
    class ResBlock(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.fc1 = nn.Linear(dim, dim)
            self.fc2 = nn.Linear(dim, dim)
            self.norm = nn.LayerNorm(dim)

        def forward(self, x):
            return x + self.fc2(F.gelu(self.fc1(self.norm(x))))

    class ResMLP(nn.Module):
        def __init__(self, in_dim=64, hidden=128, n_blocks=3, out_dim=10):
            super().__init__()
            self.proj = nn.Linear(in_dim, hidden)
            self.blocks = nn.ModuleList([ResBlock(hidden) for _ in range(n_blocks)])
            self.head = nn.Linear(hidden, out_dim)

        def forward(self, x):
            x = self.proj(x)
            for b in self.blocks:
                x = b(x)
            return self.head(x)

    model = ResMLP()
    x = torch.randn(8, 64)
    targets = torch.randint(0, 10, (8,))
    return {
        "model": model,
        "sample_args": (x,),
        "loss_fn": lambda out: F.cross_entropy(out, targets),
    }


# 7. Autoencoder (reconstruction loss)
def autoencoder_recipe():
    class Autoencoder(nn.Module):
        def __init__(self, in_dim=784, latent=32):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(in_dim, 256), nn.ReLU(),
                nn.Linear(256, latent),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent, 256), nn.ReLU(),
                nn.Linear(256, in_dim), nn.Sigmoid(),
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

    model = Autoencoder()
    x = torch.randn(8, 784).clamp(0, 1)
    return {
        "model": model,
        "sample_args": (x,),
        "loss_fn": lambda out: F.mse_loss(out, x),
    }


# 8. Multi-head attention standalone
def mha_recipe():
    class MHABlock(nn.Module):
        def __init__(self, d_model=64, nhead=4):
            super().__init__()
            self.mha = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=0.0)
            self.norm = nn.LayerNorm(d_model)
            self.fc = nn.Linear(d_model, 10)

        def forward(self, x):
            h, _ = self.mha(x, x, x)
            h = self.norm(h + x)
            return self.fc(h.mean(dim=1))

    model = MHABlock()
    x = torch.randn(4, 16, 64)
    targets = torch.randint(0, 10, (4,))
    return {
        "model": model,
        "sample_args": (x,),
        "loss_fn": lambda out: F.cross_entropy(out, targets),
    }


# 9. 1D Conv for time series
def conv1d_recipe():
    class Conv1DNet(nn.Module):
        def __init__(self, in_ch=3, seq_len=64):
            super().__init__()
            self.conv1 = nn.Conv1d(in_ch, 16, 5, padding=2)
            self.conv2 = nn.Conv1d(16, 32, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(32, 5)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.pool(x).squeeze(-1)
            return self.fc(x)

    model = Conv1DNet()
    x = torch.randn(4, 3, 64)
    targets = torch.randint(0, 5, (4,))
    return {
        "model": model,
        "sample_args": (x,),
        "loss_fn": lambda out: F.cross_entropy(out, targets),
    }


# 10. Depthwise separable convolution (MobileNet-style)
def depthwise_sep_recipe():
    class DepthwiseSep(nn.Module):
        def __init__(self, in_ch=3, mid_ch=32, out_ch=64):
            super().__init__()
            self.dw = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch)
            self.pw = nn.Conv2d(in_ch, mid_ch, 1)
            self.bn = nn.BatchNorm2d(mid_ch)
            self.dw2 = nn.Conv2d(mid_ch, mid_ch, 3, padding=1, groups=mid_ch)
            self.pw2 = nn.Conv2d(mid_ch, out_ch, 1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(out_ch, 10)

        def forward(self, x):
            x = F.relu(self.bn(self.pw(self.dw(x))))
            x = F.relu(self.pw2(self.dw2(x)))
            x = self.pool(x).flatten(1)
            return self.fc(x)

    model = DepthwiseSep()
    x = torch.randn(4, 3, 32, 32)
    targets = torch.randint(0, 10, (4,))
    return {
        "model": model,
        "sample_args": (x,),
        "loss_fn": lambda out: F.cross_entropy(out, targets),
    }


# 11. Transformer decoder (causal LM) - uses manual SDPA like real GPT impls
def causal_lm_recipe():
    class CausalSelfAttention(nn.Module):
        def __init__(self, d_model, nhead):
            super().__init__()
            self.n_head = nhead
            self.head_dim = d_model // nhead
            self.qkv = nn.Linear(d_model, 3 * d_model)
            self.proj = nn.Linear(d_model, d_model)

        def forward(self, x):
            B, T, C = x.shape
            qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim)
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            return self.proj(y.transpose(1, 2).reshape(B, T, C))

    class Block(nn.Module):
        def __init__(self, d_model, nhead, ff_dim):
            super().__init__()
            self.ln1 = nn.LayerNorm(d_model)
            self.attn = CausalSelfAttention(d_model, nhead)
            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, ff_dim), nn.GELU(),
                nn.Linear(ff_dim, d_model),
            )

        def forward(self, x):
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
            return x

    class TinyGPT(nn.Module):
        def __init__(self, vocab_size=128, d_model=64, nhead=4, n_layers=2, max_len=64):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(max_len, d_model)
            self.blocks = nn.ModuleList([Block(d_model, nhead, 128) for _ in range(n_layers)])
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size)

        def forward(self, x):
            B, T = x.shape
            pos = torch.arange(T, device=x.device)
            h = self.tok_emb(x) + self.pos_emb(pos)
            for block in self.blocks:
                h = block(h)
            return self.head(self.ln_f(h))

    model = TinyGPT()
    x = torch.randint(0, 128, (4, 32))
    targets = torch.randint(0, 128, (4, 32))
    return {
        "model": model,
        "sample_args": (x,),
        "loss_fn": lambda out: F.cross_entropy(out.view(-1, 128), targets.view(-1)),
    }


# 12. Variational Autoencoder (multi-output, reparameterization)
def vae_recipe():
    class VAE(nn.Module):
        def __init__(self, in_dim=784, latent=16):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU())
            self.fc_mu = nn.Linear(128, latent)
            self.fc_logvar = nn.Linear(128, latent)
            self.dec = nn.Sequential(
                nn.Linear(latent, 128), nn.ReLU(),
                nn.Linear(128, in_dim), nn.Sigmoid(),
            )
            self._in_dim = in_dim

        def forward(self, x):
            h = self.enc(x)
            mu, logvar = self.fc_mu(h), self.fc_logvar(h)
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
            recon = self.dec(z)
            # VAE loss
            recon_loss = F.binary_cross_entropy(recon, x, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            return (recon_loss + kl_loss) / x.size(0)

    model = VAE()
    x = torch.rand(8, 784)  # binary-ish inputs
    return {
        "model": model,
        "sample_args": (x,),
        "loss_fn": None,  # model already returns scalar loss
    }


# 13. Mixture of Experts (gating network)
def moe_recipe():
    class Expert(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.fc = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        def forward(self, x):
            return self.fc(x)

    class MoE(nn.Module):
        def __init__(self, dim=64, n_experts=4):
            super().__init__()
            self.experts = nn.ModuleList([Expert(dim) for _ in range(n_experts)])
            self.gate = nn.Linear(dim, n_experts)
            self.head = nn.Linear(dim, 10)

        def forward(self, x):
            gate_logits = self.gate(x)  # (B, n_experts)
            weights = F.softmax(gate_logits, dim=-1)  # (B, n_experts)
            expert_outs = torch.stack([e(x) for e in self.experts], dim=1)  # (B, n_experts, dim)
            mixed = (weights.unsqueeze(-1) * expert_outs).sum(dim=1)  # (B, dim)
            return self.head(mixed)

    model = MoE()
    x = torch.randn(8, 64)
    targets = torch.randint(0, 10, (8,))
    return {
        "model": model,
        "sample_args": (x,),
        "loss_fn": lambda out: F.cross_entropy(out, targets),
    }


# 14. Squeeze-and-Excitation block
def se_recipe():
    class SEBlock(nn.Module):
        def __init__(self, ch, reduction=4):
            super().__init__()
            self.fc1 = nn.Linear(ch, ch // reduction)
            self.fc2 = nn.Linear(ch // reduction, ch)

        def forward(self, x):
            # x: (B, C, H, W)
            s = x.mean(dim=(2, 3))  # (B, C)
            s = F.relu(self.fc1(s))
            s = torch.sigmoid(self.fc2(s))
            return x * s.unsqueeze(-1).unsqueeze(-1)

    class SENet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3, padding=1)
            self.se = SEBlock(16)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(16, 10)

        def forward(self, x):
            x = F.relu(self.conv(x))
            x = self.se(x)
            x = self.pool(x).flatten(1)
            return self.fc(x)

    model = SENet()
    x = torch.randn(4, 3, 16, 16)
    targets = torch.randint(0, 10, (4,))
    return {
        "model": model,
        "sample_args": (x,),
        "loss_fn": lambda out: F.cross_entropy(out, targets),
    }


# 15. GroupNorm + InstanceNorm model
def norm_variety_recipe():
    class NormModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.gn = nn.GroupNorm(4, 16)
            self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
            self.ins_norm = nn.InstanceNorm2d(16)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(16, 5)

        def forward(self, x):
            x = F.relu(self.gn(self.conv1(x)))
            x = F.relu(self.ins_norm(self.conv2(x)))
            x = self.pool(x).flatten(1)
            return self.fc(x)

    model = NormModel()
    x = torch.randn(4, 3, 16, 16)
    targets = torch.randint(0, 5, (4,))
    return {
        "model": model,
        "sample_args": (x,),
        "loss_fn": lambda out: F.cross_entropy(out, targets),
    }


# 16. U-Net style encoder-decoder with skip connections
def unet_recipe():
    class TinyUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU())
            self.enc2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.ReLU())
            self.dec2 = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1), nn.ReLU())
            self.dec1 = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1), nn.ReLU())  # 32 = skip
            self.head = nn.Conv2d(16, 1, 1)

        def forward(self, x):
            e1 = self.enc1(x)                   # (B, 16, H, W)
            e2 = self.enc2(F.max_pool2d(e1, 2)) # (B, 32, H/2, W/2)
            d2 = self.dec2(e2)                   # (B, 16, H/2, W/2)
            d2_up = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
            d1 = self.dec1(torch.cat([d2_up, e1], dim=1))  # skip
            return self.head(d1)

    model = TinyUNet()
    x = torch.randn(2, 1, 32, 32)
    target = torch.randn(2, 1, 32, 32)
    return {
        "model": model,
        "sample_args": (x,),
        "loss_fn": lambda out: F.mse_loss(out, target),
    }


# 17. Multi-input model (two-branch)
def multi_input_recipe():
    class TwoBranch(nn.Module):
        def __init__(self):
            super().__init__()
            self.branch_a = nn.Sequential(nn.Linear(32, 64), nn.ReLU())
            self.branch_b = nn.Sequential(nn.Linear(48, 64), nn.ReLU())
            self.head = nn.Linear(128, 10)

        def forward(self, a, b):
            ha = self.branch_a(a)
            hb = self.branch_b(b)
            return self.head(torch.cat([ha, hb], dim=-1))

    model = TwoBranch()
    a = torch.randn(4, 32)
    b = torch.randn(4, 48)
    targets = torch.randint(0, 10, (4,))
    return {
        "model": model,
        "sample_args": (a, b),
        "loss_fn": lambda out: F.cross_entropy(out, targets),
    }


# 18. Dropout-heavy model (training mode matters)
def dropout_recipe():
    class DropoutModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(64, 128)
            self.drop1 = nn.Dropout(0.5)
            self.fc2 = nn.Linear(128, 128)
            self.drop2 = nn.Dropout(0.3)
            self.fc3 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.drop1(F.relu(self.fc1(x)))
            x = self.drop2(F.relu(self.fc2(x)))
            return self.fc3(x)

    model = DropoutModel()
    model.train()
    x = torch.randn(8, 64)
    targets = torch.randint(0, 10, (8,))
    return {
        "model": model,
        "sample_args": (x,),
        "loss_fn": lambda out: F.cross_entropy(out, targets),
    }


# 19. Weight sharing (tied embeddings, like GPT)
def tied_weights_recipe():
    class TiedLM(nn.Module):
        def __init__(self, vocab_size=100, dim=64):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, dim)
            self.fc = nn.Linear(dim, dim)
            # Tie weights
            self.head = nn.Linear(dim, vocab_size, bias=False)
            self.head.weight = self.embed.weight  # weight tying

        def forward(self, x):
            h = self.embed(x)
            h = F.relu(self.fc(h))
            return self.head(h)

    model = TiedLM()
    x = torch.randint(0, 100, (4, 16))
    targets = torch.randint(0, 100, (4, 16))
    return {
        "model": model,
        "sample_args": (x,),
        "loss_fn": lambda out: F.cross_entropy(out.view(-1, 100), targets.view(-1)),
    }


# 20. Spectral/complex ops: softmax, log_softmax, sigmoid, tanh mix
def activation_zoo_recipe():
    class ActivationZoo(nn.Module):
        def __init__(self, dim=64):
            super().__init__()
            self.fc1 = nn.Linear(dim, dim)
            self.fc2 = nn.Linear(dim, dim)
            self.fc3 = nn.Linear(dim, dim)
            self.fc4 = nn.Linear(dim, dim)
            self.head = nn.Linear(dim * 4, 10)

        def forward(self, x):
            a = torch.sigmoid(self.fc1(x))
            b = torch.tanh(self.fc2(x))
            c = F.silu(self.fc3(x))
            d = F.gelu(self.fc4(x))
            return self.head(torch.cat([a, b, c, d], dim=-1))

    model = ActivationZoo()
    x = torch.randn(4, 64)
    targets = torch.randint(0, 10, (4,))
    return {
        "model": model,
        "sample_args": (x,),
        "loss_fn": lambda out: F.cross_entropy(out, targets),
    }


# 21. Bidirectional LSTM
def bilstm_recipe():
    class BiLSTM(nn.Module):
        def __init__(self, vocab=80, emb=32, hidden=48):
            super().__init__()
            self.embed = nn.Embedding(vocab, emb)
            self.lstm = nn.LSTM(emb, hidden, bidirectional=True, batch_first=True)
            self.fc = nn.Linear(hidden * 2, 5)

        def forward(self, x):
            h = self.embed(x)
            out, _ = self.lstm(h)
            return self.fc(out[:, -1, :])

    model = BiLSTM()
    x = torch.randint(0, 80, (4, 20))
    targets = torch.randint(0, 5, (4,))
    return {
        "model": model,
        "sample_args": (x,),
        "loss_fn": lambda out: F.cross_entropy(out, targets),
    }


# 22. Pixel Shuffle (sub-pixel convolution, super-resolution style)
def pixelshuffle_recipe():
    class SuperRes(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 12, 3, padding=1)  # 12 = 3 * 2^2
            self.ps = nn.PixelShuffle(2)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = self.conv2(x)
            return self.ps(x)

    model = SuperRes()
    lr = torch.randn(2, 3, 8, 8)
    hr = torch.randn(2, 3, 16, 16)
    return {
        "model": model,
        "sample_args": (lr,),
        "loss_fn": lambda out: F.mse_loss(out, hr),
    }


# 23. Multiple loss heads (auxiliary loss)
def multi_loss_recipe():
    class MultiHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(nn.Linear(64, 128), nn.ReLU())
            self.head1 = nn.Linear(128, 10)
            self.head2 = nn.Linear(128, 5)

        def forward(self, x):
            h = self.backbone(x)
            return self.head1(h), self.head2(h)

    model = MultiHead()
    x = torch.randn(4, 64)
    t1 = torch.randint(0, 10, (4,))
    t2 = torch.randint(0, 5, (4,))
    return {
        "model": model,
        "sample_args": (x,),
        "loss_fn": lambda out: F.cross_entropy(out[0], t1) + F.cross_entropy(out[1], t2),
    }


# 24. Embedding bag (sparse gradients)
def embeddingbag_recipe():
    class BagModel(nn.Module):
        def __init__(self, n_items=500, dim=32):
            super().__init__()
            self.bag = nn.EmbeddingBag(n_items, dim, mode='mean')
            self.fc = nn.Linear(dim, 1)

        def forward(self, x, offsets):
            e = self.bag(x, offsets)
            return self.fc(e).squeeze(-1)

    model = BagModel()
    x = torch.randint(0, 500, (20,))
    offsets = torch.tensor([0, 5, 10, 15])
    targets = torch.randn(4)
    return {
        "model": model,
        "sample_args": (x, offsets),
        "loss_fn": lambda out: F.mse_loss(out, targets),
    }


# 25. Dilated convolution (WaveNet-style)
def dilated_conv_recipe():
    class DilatedNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.convs = nn.ModuleList([
                nn.Conv1d(16, 16, 3, dilation=2**i, padding=2**i)
                for i in range(4)
            ])
            self.proj_in = nn.Conv1d(1, 16, 1)
            self.proj_out = nn.Conv1d(16, 1, 1)

        def forward(self, x):
            x = self.proj_in(x)
            for conv in self.convs:
                residual = x
                x = torch.tanh(conv(x)) + residual
            return self.proj_out(x)

    model = DilatedNet()
    x = torch.randn(4, 1, 64)
    target = torch.randn(4, 1, 64)
    return {
        "model": model,
        "sample_args": (x,),
        "loss_fn": lambda out: F.mse_loss(out, target),
    }


# 26. Deep residual convnet (ResNet-like)
def resnet_like_recipe():
    class ResBlock2d(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(ch)
            self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(ch)

        def forward(self, x):
            h = F.relu(self.bn1(self.conv1(x)))
            h = self.bn2(self.conv2(h))
            return F.relu(h + x)

    class TinyResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1, bias=False),
                nn.BatchNorm2d(16), nn.ReLU(),
            )
            self.blocks = nn.Sequential(*[ResBlock2d(16) for _ in range(4)])
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(16, 10)

        def forward(self, x):
            x = self.stem(x)
            x = self.blocks(x)
            return self.fc(self.pool(x).flatten(1))

    model = TinyResNet()
    x = torch.randn(4, 3, 32, 32)
    targets = torch.randint(0, 10, (4,))
    return {
        "model": model,
        "sample_args": (x,),
        "loss_fn": lambda out: F.cross_entropy(out, targets),
    }


# 27. Multi-scale feature extraction (FPN-like)
def multiscale_recipe():
    class MultiScale(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_s1 = nn.Conv2d(3, 8, 3, padding=1)
            self.conv_s2 = nn.Conv2d(8, 16, 3, stride=2, padding=1)
            self.conv_s4 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
            self.up2 = nn.ConvTranspose2d(32, 16, 2, stride=2)
            self.up1 = nn.ConvTranspose2d(16, 8, 2, stride=2)
            self.head = nn.Conv2d(8, 1, 1)

        def forward(self, x):
            s1 = F.relu(self.conv_s1(x))
            s2 = F.relu(self.conv_s2(s1))
            s4 = F.relu(self.conv_s4(s2))
            d2 = F.relu(self.up2(s4) + s2)
            d1 = F.relu(self.up1(d2) + s1)
            return self.head(d1)

    model = MultiScale()
    x = torch.randn(2, 3, 16, 16)
    target = torch.randn(2, 1, 16, 16)
    return {
        "model": model,
        "sample_args": (x,),
        "loss_fn": lambda out: F.mse_loss(out, target),
    }


# 28. Label smoothing + mixup-style computation
def smooth_loss_recipe():
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 10)
            )

        def forward(self, x):
            return self.net(x)

    model = SimpleNet()
    x = torch.randn(8, 32)
    targets = torch.randint(0, 10, (8,))

    def label_smooth_loss(logits, eps=0.1):
        n_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        nll = F.nll_loss(log_probs, targets, reduction='mean')
        smooth = -log_probs.mean(dim=-1).mean()
        return (1 - eps) * nll + eps * smooth

    return {
        "model": model,
        "sample_args": (x,),
        "loss_fn": label_smooth_loss,
    }


# 29. Transformer with RoPE (rotary position embedding)
def rope_recipe():
    class RoPEAttention(nn.Module):
        def __init__(self, dim=64, nhead=4):
            super().__init__()
            self.nhead = nhead
            self.head_dim = dim // nhead
            self.qkv = nn.Linear(dim, 3 * dim, bias=False)
            self.proj = nn.Linear(dim, dim)

        def _apply_rope(self, x, seq_len):
            d = x.size(-1)
            pos = torch.arange(seq_len, device=x.device).unsqueeze(1)
            dim_idx = torch.arange(0, d, 2, device=x.device).float()
            freq = 1.0 / (10000.0 ** (dim_idx / d))
            angles = pos * freq
            cos = torch.cos(angles)
            sin = torch.sin(angles)
            x1 = x[..., ::2]
            x2 = x[..., 1::2]
            return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)

        def forward(self, x):
            B, T, C = x.shape
            qkv = self.qkv(x).reshape(B, T, 3, self.nhead, self.head_dim)
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # (B, H, T, D)
            q = self._apply_rope(q, T)
            k = self._apply_rope(k, T)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            return self.proj(y.transpose(1, 2).reshape(B, T, C))

    class RoPETransformer(nn.Module):
        def __init__(self, vocab=200, dim=64, nhead=4, n_layers=2):
            super().__init__()
            self.embed = nn.Embedding(vocab, dim)
            self.layers = nn.ModuleList([
                nn.ModuleDict({
                    'attn': RoPEAttention(dim, nhead),
                    'ln1': nn.LayerNorm(dim),
                    'ff': nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim)),
                    'ln2': nn.LayerNorm(dim),
                })
                for _ in range(n_layers)
            ])
            self.head = nn.Linear(dim, vocab)

        def forward(self, x):
            h = self.embed(x)
            for layer in self.layers:
                h = h + layer['attn'](layer['ln1'](h))
                h = h + layer['ff'](layer['ln2'](h))
            return self.head(h)

    model = RoPETransformer()
    x = torch.randint(0, 200, (4, 32))
    targets = torch.randint(0, 200, (4, 32))
    return {
        "model": model,
        "sample_args": (x,),
        "loss_fn": lambda out: F.cross_entropy(out.view(-1, 200), targets.view(-1)),
    }


# 30. Contrastive learning model (two-tower + cosine similarity)
def contrastive_recipe():
    class Encoder(nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 128), nn.ReLU(),
                nn.Linear(128, out_dim),
            )
        def forward(self, x):
            h = self.net(x)
            return F.normalize(h, dim=-1)

    class ContrastiveModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc_a = Encoder(64, 32)
            self.enc_b = Encoder(48, 32)
            self.temp = nn.Parameter(torch.tensor(0.07))

        def forward(self, a, b):
            za = self.enc_a(a)
            zb = self.enc_b(b)
            logits = za @ zb.T / self.temp.exp()
            return logits

    model = ContrastiveModel()
    a = torch.randn(8, 64)
    b = torch.randn(8, 48)
    targets = torch.arange(8)  # diagonal is positive pair
    return {
        "model": model,
        "sample_args": (a, b),
        "loss_fn": lambda logits: F.cross_entropy(logits, targets),
    }


# ═══════════════════════════════════════════════════════════════════════
# Test runner
# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
# Function-wrapping test cases (use extract_function instead of
# extract_training_step)
# ═══════════════════════════════════════════════════════════════════════

def fn_plain_function():
    """Plain function (not a module) with closure over weights."""
    w = torch.randn(16, 32)
    b = torch.randn(16)
    def my_fn(x):
        return F.relu(x @ w.T + b)
    return {"fn": my_fn, "args": (torch.randn(4, 32),), "run_backward": True,
            "loss_fn": lambda out: out.sum()}


def fn_partial_model():
    """Capture only one layer of a larger model."""
    class BigModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Sequential(nn.Linear(64, 64), nn.ReLU()) for _ in range(10)
            ])
            self.head = nn.Linear(64, 10)
        def forward(self, x):
            for l in self.layers:
                x = l(x)
            return self.head(x)
    model = BigModel()
    # Only capture the first layer
    return {"fn": model.layers[0], "args": (torch.randn(4, 64),),
            "run_backward": True, "loss_fn": lambda out: out.sum()}


def fn_custom_attention():
    """Custom SDPA-based attention as a free function."""
    wq = torch.randn(64, 64)
    wk = torch.randn(64, 64)
    wv = torch.randn(64, 64)
    def attention(x):
        B, T, C = x.shape
        q = (x @ wq).view(B, T, 4, 16).transpose(1, 2)
        k = (x @ wk).view(B, T, 4, 16).transpose(1, 2)
        v = (x @ wv).view(B, T, 4, 16).transpose(1, 2)
        return F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, C)
    return {"fn": attention, "args": (torch.randn(2, 16, 64),),
            "run_backward": True, "loss_fn": lambda out: out.sum()}


def fn_forward_only():
    """Forward-only capture (no backward)."""
    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 10))
    return {"fn": model, "args": (torch.randn(4, 32),), "run_backward": False}


def fn_lstm_standalone():
    """LSTM layer standalone (triggers export fallback)."""
    lstm = nn.LSTM(32, 64, batch_first=True)
    return {"fn": lstm, "args": (torch.randn(4, 16, 32),),
            "run_backward": True, "loss_fn": lambda out: out[0].sum()}


def fn_loss_computation():
    """Capture just the loss computation."""
    def focal_loss(logits, targets, gamma=2.0):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** gamma * ce).mean()

    logits = torch.randn(8, 10, requires_grad=True)
    targets = torch.randint(0, 10, (8,))
    def fn(logits):
        return focal_loss(logits, targets)
    return {"fn": fn, "args": (logits,), "run_backward": True}


# All function-wrapping recipes
FN_RECIPES = {
    "fn_plain_function": fn_plain_function,
    "fn_partial_model": fn_partial_model,
    "fn_custom_attention": fn_custom_attention,
    "fn_forward_only": fn_forward_only,
    "fn_lstm_standalone": fn_lstm_standalone,
    "fn_loss_computation": fn_loss_computation,
}


ALL_RECIPES = {
    "mlp": mlp_recipe,
    "convnet": convnet_recipe,
    "lstm": lstm_recipe,
    "gru": gru_recipe,
    "transformer_encoder": transformer_encoder_recipe,
    "residual_mlp": residual_mlp_recipe,
    "autoencoder": autoencoder_recipe,
    "mha": mha_recipe,
    "conv1d": conv1d_recipe,
    "depthwise_sep": depthwise_sep_recipe,
    "causal_lm": causal_lm_recipe,
    "vae": vae_recipe,
    "moe": moe_recipe,
    "se_block": se_recipe,
    "norm_variety": norm_variety_recipe,
    "unet": unet_recipe,
    "multi_input": multi_input_recipe,
    "dropout": dropout_recipe,
    "tied_weights": tied_weights_recipe,
    "activation_zoo": activation_zoo_recipe,
    "bilstm": bilstm_recipe,
    "pixelshuffle": pixelshuffle_recipe,
    "multi_loss": multi_loss_recipe,
    "embeddingbag": embeddingbag_recipe,
    "dilated_conv": dilated_conv_recipe,
    "resnet_like": resnet_like_recipe,
    "multiscale": multiscale_recipe,
    "smooth_loss": smooth_loss_recipe,
    "rope": rope_recipe,
    "contrastive": contrastive_recipe,
}

# ═══════════════════════════════════════════════════════════════════════
# Wave 2: More challenging models
# ═══════════════════════════════════════════════════════════════════════


# 31. DenseNet-style dense connections
def densenet_recipe():
    class DenseBlock(nn.Module):
        def __init__(self, in_ch, growth=8, n_layers=4):
            super().__init__()
            self.layers = nn.ModuleList()
            for i in range(n_layers):
                self.layers.append(nn.Sequential(
                    nn.BatchNorm2d(in_ch + i * growth),
                    nn.ReLU(),
                    nn.Conv2d(in_ch + i * growth, growth, 3, padding=1),
                ))

        def forward(self, x):
            features = [x]
            for layer in self.layers:
                h = layer(torch.cat(features, dim=1))
                features.append(h)
            return torch.cat(features, dim=1)

    class TinyDenseNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Conv2d(3, 16, 3, padding=1)
            self.dense = DenseBlock(16, growth=8, n_layers=4)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(16 + 4*8, 10)  # 16 + 32 = 48

        def forward(self, x):
            x = F.relu(self.stem(x))
            x = self.dense(x)
            return self.fc(self.pool(x).flatten(1))

    model = TinyDenseNet()
    x = torch.randn(4, 3, 16, 16)
    targets = torch.randint(0, 10, (4,))
    return {
        "model": model,
        "sample_args": (x,),
        "loss_fn": lambda out: F.cross_entropy(out, targets),
    }


# 32. Knowledge distillation setup (teacher-student)
def distillation_recipe():
    class Teacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(64, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, 10),
            )
        def forward(self, x):
            return self.net(x)

    class Student(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, 10),
            )
        def forward(self, x):
            return self.net(x)

    teacher = Teacher()
    teacher.eval()
    student = Student()
    student.train()

    x = torch.randn(8, 64)
    targets = torch.randint(0, 10, (8,))

    # Distillation: KL divergence between student and teacher
    class DistillModel(nn.Module):
        def __init__(self, student, teacher):
            super().__init__()
            self.student = student
            self.teacher = teacher
        def forward(self, x):
            return self.student(x), self.teacher(x)

    model = DistillModel(student, teacher)
    T = 4.0  # temperature
    alpha = 0.7

    def distill_loss(outputs):
        s_logits, t_logits = outputs
        hard_loss = F.cross_entropy(s_logits, targets)
        soft_loss = F.kl_div(
            F.log_softmax(s_logits / T, dim=-1),
            F.softmax(t_logits / T, dim=-1),
            reduction='batchmean',
        ) * (T * T)
        return alpha * soft_loss + (1 - alpha) * hard_loss

    return {
        "model": model,
        "sample_args": (x,),
        "loss_fn": distill_loss,
    }


# 33. Transformer with SwiGLU activation (LLaMA-style FFN)
def swiglu_recipe():
    class SwiGLUFFN(nn.Module):
        def __init__(self, dim, hidden_dim):
            super().__init__()
            self.w1 = nn.Linear(dim, hidden_dim, bias=False)
            self.w2 = nn.Linear(hidden_dim, dim, bias=False)
            self.w3 = nn.Linear(dim, hidden_dim, bias=False)

        def forward(self, x):
            return self.w2(F.silu(self.w1(x)) * self.w3(x))

    class SwiGLUBlock(nn.Module):
        def __init__(self, dim=64, nhead=4, ff_mult=4):
            super().__init__()
            self.ln1 = nn.LayerNorm(dim)
            self.attn_qkv = nn.Linear(dim, 3*dim, bias=False)
            self.attn_proj = nn.Linear(dim, dim, bias=False)
            self.ln2 = nn.LayerNorm(dim)
            self.ffn = SwiGLUFFN(dim, dim * ff_mult)
            self.nhead = nhead
            self.head_dim = dim // nhead

        def forward(self, x):
            B, T, C = x.shape
            h = self.ln1(x)
            qkv = self.attn_qkv(h).reshape(B, T, 3, self.nhead, self.head_dim)
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            attn_out = attn_out.transpose(1, 2).reshape(B, T, C)
            x = x + self.attn_proj(attn_out)
            x = x + self.ffn(self.ln2(x))
            return x

    class SwiGLUTransformer(nn.Module):
        def __init__(self, vocab=200, dim=64, nhead=4, n_layers=2):
            super().__init__()
            self.embed = nn.Embedding(vocab, dim)
            self.blocks = nn.ModuleList([SwiGLUBlock(dim, nhead) for _ in range(n_layers)])
            self.ln = nn.LayerNorm(dim)
            self.head = nn.Linear(dim, vocab, bias=False)

        def forward(self, x):
            h = self.embed(x)
            for block in self.blocks:
                h = block(h)
            return self.head(self.ln(h))

    model = SwiGLUTransformer()
    x = torch.randint(0, 200, (4, 32))
    targets = torch.randint(0, 200, (4, 32))
    return {
        "model": model,
        "sample_args": (x,),
        "loss_fn": lambda out: F.cross_entropy(out.view(-1, 200), targets.view(-1)),
    }


# 34. Graph neural network (message passing)
def gnn_recipe():
    class SimpleGNN(nn.Module):
        def __init__(self, in_dim=16, hidden=32, out_dim=4, n_layers=3):
            super().__init__()
            self.embed = nn.Linear(in_dim, hidden)
            self.convs = nn.ModuleList()
            for _ in range(n_layers):
                self.convs.append(nn.ModuleDict({
                    'msg': nn.Linear(hidden, hidden),
                    'upd': nn.Linear(hidden * 2, hidden),
                }))
            self.head = nn.Linear(hidden, out_dim)

        def forward(self, x, adj):
            # x: (N, in_dim), adj: (N, N) adjacency
            h = F.relu(self.embed(x))
            for conv in self.convs:
                msg = conv['msg'](h)
                agg = adj @ msg  # simple sum aggregation
                h = F.relu(conv['upd'](torch.cat([h, agg], dim=-1)))
            return self.head(h)

    n_nodes = 20
    model = SimpleGNN()
    x = torch.randn(n_nodes, 16)
    # Random sparse adjacency (normalized)
    adj = (torch.rand(n_nodes, n_nodes) > 0.7).float()
    adj = adj / (adj.sum(dim=1, keepdim=True) + 1e-8)
    targets = torch.randint(0, 4, (n_nodes,))
    return {
        "model": model,
        "sample_args": (x, adj),
        "loss_fn": lambda out: F.cross_entropy(out, targets),
    }


# 35. Batch renormalization model
def batch_renorm_recipe():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3, padding=1)
            self.bn = nn.BatchNorm2d(16, affine=True, track_running_stats=True)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(16, 10)

        def forward(self, x):
            x = self.bn(self.conv(x))
            x = F.relu(x)
            return self.fc(self.pool(x).flatten(1))

    model = Model()
    model.train()
    x = torch.randn(8, 3, 8, 8)
    targets = torch.randint(0, 10, (8,))
    return {
        "model": model,
        "sample_args": (x,),
        "loss_fn": lambda out: F.cross_entropy(out, targets),
    }


# 36. Multi-task learning (shared backbone, multiple heads)
def multitask_recipe():
    class MultiTaskNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(4),
            )
            self.classify_head = nn.Linear(32 * 4 * 4, 10)
            self.regress_head = nn.Linear(32 * 4 * 4, 3)

        def forward(self, x):
            features = self.backbone(x).flatten(1)
            cls = self.classify_head(features)
            reg = self.regress_head(features)
            return cls, reg

    model = MultiTaskNet()
    x = torch.randn(4, 3, 16, 16)
    cls_targets = torch.randint(0, 10, (4,))
    reg_targets = torch.randn(4, 3)
    return {
        "model": model,
        "sample_args": (x,),
        "loss_fn": lambda out: F.cross_entropy(out[0], cls_targets) + F.mse_loss(out[1], reg_targets),
    }


# 37. Deformable convolution-like (grid sampling)
def grid_sample_recipe():
    class SpatialTransform(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3, padding=1)
            self.offset = nn.Conv2d(16, 2, 3, padding=1)  # predict x,y offsets
            self.final = nn.Conv2d(16, 10, 1)
            self.pool = nn.AdaptiveAvgPool2d(1)

        def forward(self, x):
            feat = F.relu(self.conv(x))
            B, C, H, W = feat.shape
            # Generate sampling grid
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(-1, 1, H, device=x.device),
                torch.linspace(-1, 1, W, device=x.device),
                indexing='ij',
            )
            grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
            offsets = self.offset(feat).permute(0, 2, 3, 1) * 0.1
            grid = grid + offsets
            warped = F.grid_sample(feat, grid, align_corners=True, mode='bilinear')
            return self.pool(self.final(warped)).flatten(1)

    model = SpatialTransform()
    x = torch.randn(4, 3, 16, 16)
    targets = torch.randint(0, 10, (4,))
    return {
        "model": model,
        "sample_args": (x,),
        "loss_fn": lambda out: F.cross_entropy(out, targets),
    }


# 38. Mixture density network (output is distribution params)
def mdn_recipe():
    class MDN(nn.Module):
        def __init__(self, in_dim=8, hidden=64, n_mix=5, out_dim=2):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
            )
            self.pi = nn.Linear(hidden, n_mix)
            self.mu = nn.Linear(hidden, n_mix * out_dim)
            self.sigma = nn.Linear(hidden, n_mix * out_dim)
            self.n_mix = n_mix
            self.out_dim = out_dim

        def forward(self, x):
            h = self.backbone(x)
            pi = F.softmax(self.pi(h), dim=-1)
            mu = self.mu(h).view(-1, self.n_mix, self.out_dim)
            sigma = torch.exp(self.sigma(h)).view(-1, self.n_mix, self.out_dim)
            return pi, mu, sigma

    model = MDN()
    x = torch.randn(16, 8)
    targets = torch.randn(16, 2)

    def mdn_loss(outputs):
        pi, mu, sigma = outputs
        # Negative log-likelihood
        target = targets.unsqueeze(1)  # (B, 1, out_dim)
        normal = torch.exp(-0.5 * ((target - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))
        prob = (pi.unsqueeze(-1) * normal).sum(dim=1).prod(dim=-1)
        return -torch.log(prob + 1e-8).mean()

    return {
        "model": model,
        "sample_args": (x,),
        "loss_fn": mdn_loss,
    }


# 39. Conditional batch norm (class-conditional generation)
def cond_bn_recipe():
    class CondBN(nn.Module):
        def __init__(self, ch, n_classes=10):
            super().__init__()
            self.bn = nn.BatchNorm2d(ch, affine=False)
            self.gamma = nn.Embedding(n_classes, ch)
            self.beta = nn.Embedding(n_classes, ch)
            nn.init.ones_(self.gamma.weight)
            nn.init.zeros_(self.beta.weight)

        def forward(self, x, y):
            h = self.bn(x)
            gamma = self.gamma(y).unsqueeze(-1).unsqueeze(-1)
            beta = self.beta(y).unsqueeze(-1).unsqueeze(-1)
            return gamma * h + beta

    class CondNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3, padding=1)
            self.cbn = CondBN(16)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(16, 10)

        def forward(self, x, labels):
            h = self.conv(x)
            h = F.relu(self.cbn(h, labels))
            return self.fc(self.pool(h).flatten(1))

    model = CondNet()
    x = torch.randn(4, 3, 16, 16)
    labels = torch.randint(0, 10, (4,))
    targets = torch.randint(0, 10, (4,))
    return {
        "model": model,
        "sample_args": (x, labels),
        "loss_fn": lambda out: F.cross_entropy(out, targets),
    }


# 40. Spectral normalization
def spectral_norm_recipe():
    model = nn.Sequential(
        nn.utils.parametrizations.spectral_norm(nn.Linear(64, 128)),
        nn.ReLU(),
        nn.utils.parametrizations.spectral_norm(nn.Linear(128, 128)),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    x = torch.randn(8, 64)
    targets = torch.randint(0, 10, (8,))
    return {
        "model": model,
        "sample_args": (x,),
        "loss_fn": lambda out: F.cross_entropy(out, targets),
    }


# ═══════════════════════════════════════════════════════════════════════
# Wave 3: Real open-source models
# ═══════════════════════════════════════════════════════════════════════


# 41. minGPT (karpathy)
def mingpt_recipe():
    import sys
    sys.path.insert(0, "outputs/repos/minGPT")
    from mingpt.model import GPT
    config = GPT.get_default_config()
    config.model_type = None
    config.vocab_size = 256
    config.block_size = 64
    config.n_layer = 4
    config.n_head = 4
    config.n_embd = 128
    config.embd_pdrop = 0.0
    config.resid_pdrop = 0.0
    config.attn_pdrop = 0.0
    model = GPT(config)
    x = torch.randint(0, 256, (4, 64))
    targets = torch.randint(0, 256, (4, 64))
    return {
        "model": model,
        "sample_args": (x, targets),
        "loss_fn": lambda out: out[1],  # GPT returns (logits, loss)
    }


# 42. minGPT single block (function wrapping)
def fn_mingpt_block():
    import sys
    sys.path.insert(0, "outputs/repos/minGPT")
    from mingpt.model import GPT
    config = GPT.get_default_config()
    config.model_type = None
    config.vocab_size = 256
    config.block_size = 64
    config.n_layer = 4
    config.n_head = 4
    config.n_embd = 128
    config.embd_pdrop = 0.0
    config.resid_pdrop = 0.0
    config.attn_pdrop = 0.0
    model = GPT(config)
    # Extract just one transformer block
    block = model.transformer.h[0]
    h = torch.randn(4, 64, 128)
    return {"fn": block, "args": (h,), "run_backward": True,
            "loss_fn": lambda out: out.sum()}


# 43. ViT (Vision Transformer from vit-pytorch)
def vit_recipe():
    from vit_pytorch import ViT
    model = ViT(
        image_size=32, patch_size=8, num_classes=10,
        dim=128, depth=4, heads=4, mlp_dim=256,
        dropout=0.0, emb_dropout=0.0,
    )
    x = torch.randn(4, 3, 32, 32)
    targets = torch.randint(0, 10, (4,))
    return {
        "model": model,
        "sample_args": (x,),
        "loss_fn": lambda out: F.cross_entropy(out, targets),
    }


# 44. ViT single attention layer (function wrapping)
def fn_vit_attention():
    from vit_pytorch import ViT
    model = ViT(
        image_size=32, patch_size=8, num_classes=10,
        dim=128, depth=4, heads=4, mlp_dim=256,
        dropout=0.0, emb_dropout=0.0,
    )
    # Extract first attention module
    attn = model.transformer.layers[0][0]  # [0] is Attention, [1] is FFN
    h = torch.randn(4, 17, 128)  # 16 patches + 1 CLS token
    return {"fn": attn, "args": (h,), "run_backward": True,
            "loss_fn": lambda out: out.sum()}


def _import_from_file(name, path):
    """Import a module from a file path without polluting sys.path.

    Temporarily suppresses argparse.parse_args() calls in the imported
    module (some PyTorch examples have top-level arg parsing).
    """
    import importlib.util
    import argparse

    # Temporarily neuter argparse so top-level parse_args() doesn't fail.
    # Return defaults for all registered arguments.
    _orig_parse_args = argparse.ArgumentParser.parse_args
    def _fake_parse(self, args=None, namespace=None):
        ns = argparse.Namespace()
        for action in self._actions:
            if action.default is not argparse.SUPPRESS:
                setattr(ns, action.dest, action.default)
        return ns
    argparse.ArgumentParser.parse_args = _fake_parse

    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        argparse.ArgumentParser.parse_args = _orig_parse_args

    return mod


# 45. PyTorch MNIST example (official)
def pytorch_mnist_recipe():
    mod = _import_from_file("mnist_main", "outputs/repos/pytorch-examples/mnist/main.py")
    Net = mod.Net
    model = Net()
    x = torch.randn(4, 1, 28, 28)
    targets = torch.randint(0, 10, (4,))
    return {
        "model": model,
        "sample_args": (x,),
        "loss_fn": lambda out: F.nll_loss(out, targets),
    }


# 46. PyTorch VAE example (official)
def pytorch_vae_recipe():
    mod = _import_from_file("vae_main", "outputs/repos/pytorch-examples/vae/main.py")
    VAE = mod.VAE
    model = VAE().cpu()  # force CPU (module may default to cuda)
    x = torch.rand(4, 1, 28, 28)  # [0,1] range for BCE
    def vae_loss(outputs):
        recon_x, mu, logvar = outputs
        bce = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return bce + kld
    return {
        "model": model,
        "sample_args": (x,),
        "loss_fn": vae_loss,
    }


# 47. PyTorch Word Language Model (Transformer) - official
def pytorch_wlm_recipe():
    # Need model.py on the import path for the TransformerModel's PositionalEncoding
    sys.path.insert(0, "outputs/repos/pytorch-examples/word_language_model")
    mod = _import_from_file("wlm_model", "outputs/repos/pytorch-examples/word_language_model/model.py")
    TransformerModel = mod.TransformerModel
    model = TransformerModel(ntoken=100, ninp=64, nhead=4, nhid=128, nlayers=2, dropout=0.0)
    # Input: (seq_len, batch)
    x = torch.randint(0, 100, (32, 4))
    targets = torch.randint(0, 100, (32 * 4,))
    # Pre-compute causal mask and monkey-patch forward to pass is_causal=True,
    # avoiding graph break from _detect_is_causal_mask's data-dependent bool().
    model.src_mask = model._generate_square_subsequent_mask(32)
    _orig_forward = model.forward
    def _patched_forward(src, has_mask=True):
        src = model.input_emb(src) * math.sqrt(model.ninp)
        src = model.pos_encoder(src)
        output = model.encoder(src, mask=model.src_mask, is_causal=True)
        output = model.decoder(output)
        return F.log_softmax(output, dim=-1)
    model.forward = _patched_forward
    return {
        "model": model,
        "sample_args": (x,),
        "loss_fn": lambda out: F.nll_loss(out.view(-1, 100), targets),
    }


# 48. PyTorch Super Resolution example
def pytorch_superres_recipe():
    mod = _import_from_file("superres_model", "outputs/repos/pytorch-examples/super_resolution/model.py")
    SuperResNet = mod.Net
    model = SuperResNet(upscale_factor=2)
    x = torch.randn(2, 1, 16, 16)
    target = torch.randn(2, 1, 32, 32)
    return {
        "model": model,
        "sample_args": (x,),
        "loss_fn": lambda out: F.mse_loss(out, target),
    }


# 49. PyTorch DCGAN Generator
def pytorch_dcgan_gen_recipe():
    class Generator(nn.Module):
        def __init__(self, nz=100, ngf=32, nc=3):
            super().__init__()
            self.main = nn.Sequential(
                nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 4), nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2), nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf), nn.ReLU(True),
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh(),
            )
        def forward(self, x):
            return self.main(x)

    model = Generator()
    noise = torch.randn(4, 100, 1, 1)
    target = torch.randn(4, 3, 32, 32)
    return {
        "model": model,
        "sample_args": (noise,),
        "loss_fn": lambda out: F.mse_loss(out, target),
    }


# 50. PyTorch Regression example
def pytorch_regression_recipe():
    model = nn.Linear(4, 1)
    # Polynomial features
    x = torch.randn(32, 1)
    batch_x = torch.cat([x ** i for i in range(1, 5)], dim=1)
    batch_y = torch.sin(x).squeeze()
    return {
        "model": model,
        "sample_args": (batch_x,),
        "loss_fn": lambda out: F.smooth_l1_loss(out.squeeze(), batch_y),
    }


# 51. nanochat (Karpathy) — real tokenizer + real text
def _nanochat_tokenizer():
    """Build nanochat tokenizer (GPT-2 BPE + 9 special tokens)."""
    sys.path.insert(0, "outputs/repos/nanochat")
    import tiktoken
    from nanochat.tokenizer import SPECIAL_TOKENS, RustBPETokenizer
    base = tiktoken.get_encoding("gpt2")
    special_tokens = {**base._special_tokens}
    for i, name in enumerate(SPECIAL_TOKENS):
        special_tokens[name] = base.n_vocab + i
    enc = tiktoken.Encoding(
        name="gpt2_nanochat", pat_str=base._pat_str,
        mergeable_ranks=base._mergeable_ranks, special_tokens=special_tokens,
    )
    return RustBPETokenizer(enc, "<|bos|>")


def _nanochat_model(vocab_size, seq_len=64):
    sys.path.insert(0, "outputs/repos/nanochat")
    from nanochat.gpt import GPT, GPTConfig
    config = GPTConfig(n_layer=2, n_head=4, n_kv_head=4, n_embd=128,
                       vocab_size=vocab_size, sequence_len=seq_len)
    with torch.device('cpu'):
        model = GPT(config, pad_vocab_size_to=64)
    model.init_weights()
    return model


def nanochat_recipe():
    tok = _nanochat_tokenizer()
    model = _nanochat_model(tok.get_vocab_size())
    # Real text pre-training data
    tokens = tok.encode(
        "The Pythagorean theorem states that in a right triangle, "
        "the square of the hypotenuse is equal to the sum of squares."
    )
    # Build (input, target) from contiguous token sequence
    seq_len = 32
    tokens = tokens[:seq_len + 1]
    if len(tokens) < seq_len + 1:
        tokens += [tok.get_bos_token_id()] * (seq_len + 1 - len(tokens))
    batch = torch.tensor([tokens, tokens], dtype=torch.long)
    x = batch[:, :-1].contiguous()
    targets = batch[:, 1:].contiguous()
    return {
        "model": model,
        "sample_args": (x,),
        "sample_kwargs": {"targets": targets},
    }


# 51b. nanochat SFT (real conversation masking via tokenizer)
def nanochat_sft_recipe():
    tok = _nanochat_tokenizer()
    model = _nanochat_model(tok.get_vocab_size(), seq_len=64)
    # Real conversation through render_conversation()
    conv = {
        "messages": [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
        ]
    }
    ids, mask = tok.render_conversation(conv, max_tokens=65)
    bos = tok.get_bos_token_id()
    seq_len = 32
    if len(ids) < seq_len + 1:
        ids += [bos] * (seq_len + 1 - len(ids))
        mask += [0] * (seq_len + 1 - len(mask))
    batch = torch.tensor([ids[:seq_len+1], ids[:seq_len+1]], dtype=torch.long)
    mask_t = torch.tensor([mask[:seq_len+1], mask[:seq_len+1]], dtype=torch.int8)
    x = batch[:, :-1].to(dtype=torch.int32).contiguous()
    targets = batch[:, 1:].to(dtype=torch.int64).contiguous()
    targets[mask_t[:, 1:].contiguous() == 0] = -1
    return {
        "model": model,
        "sample_args": (x,),
        "sample_kwargs": {"targets": targets},
    }


# 51c. nanochat RL (GRPO-style policy gradient with real tokens)
def nanochat_rl_recipe():
    tok = _nanochat_tokenizer()
    model = _nanochat_model(tok.get_vocab_size(), seq_len=64)
    conv = {
        "messages": [
            {"role": "user", "content": "Solve 15 * 23"},
            {"role": "assistant", "content": "15 times 23 equals 345."},
        ]
    }
    ids, mask = tok.render_conversation(conv, max_tokens=65)
    bos = tok.get_bos_token_id()
    batch_size, seq_len = 2, 32
    if len(ids) < seq_len + 1:
        ids += [bos] * (seq_len + 1 - len(ids))
        mask += [0] * (seq_len + 1 - len(mask))
    batch = torch.tensor([ids[:seq_len+1], ids[:seq_len+1]], dtype=torch.long)
    mask_t = torch.tensor([mask[:seq_len+1], mask[:seq_len+1]], dtype=torch.int8)
    x = batch[:, :-1].to(dtype=torch.int32).contiguous()
    targets = batch[:, 1:].to(dtype=torch.int64).contiguous()
    targets[mask_t[:, 1:].contiguous() == 0] = -1
    # Per-sequence advantages (reward - baseline)
    advantages = torch.tensor([1.0, -0.5])

    def rl_loss_fn(per_token_nll):
        logp = -per_token_nll.view(batch_size, seq_len)
        adv = advantages.to(logp.device).unsqueeze(-1)
        pg_obj = (logp * adv).sum()
        num_valid = (per_token_nll.to(logp.device) != 0).sum().clamp(min=1)
        return -(pg_obj / num_valid)

    return {
        "model": model,
        "sample_args": (x,),
        "sample_kwargs": {"targets": targets, "loss_reduction": "none"},
        "loss_fn": rl_loss_fn,
    }


# ═══════════════════════════════════════════════════════════════════════
# Wave 4: torchvision & library models
# ═══════════════════════════════════════════════════════════════════════

# 52. ResNet-18 (torchvision)
def resnet18_recipe():
    import torchvision.models as models
    model = models.resnet18(weights=None)
    x = torch.randn(2, 3, 32, 32)
    targets = torch.randint(0, 1000, (2,))
    return {
        "model": model,
        "sample_args": (x,),
        "loss_fn": lambda out: F.cross_entropy(out, targets),
    }


# 53. MobileNetV2 (torchvision)
def mobilenetv2_recipe():
    import torchvision.models as models
    model = models.mobilenet_v2(weights=None)
    x = torch.randn(2, 3, 32, 32)
    targets = torch.randint(0, 1000, (2,))
    return {
        "model": model,
        "sample_args": (x,),
        "loss_fn": lambda out: F.cross_entropy(out, targets),
    }


# 54. EfficientNet-B0 (torchvision)
def efficientnet_b0_recipe():
    import torchvision.models as models
    model = models.efficientnet_b0(weights=None)
    x = torch.randn(2, 3, 32, 32)
    targets = torch.randint(0, 1000, (2,))
    return {
        "model": model,
        "sample_args": (x,),
        "loss_fn": lambda out: F.cross_entropy(out, targets),
    }


# 55. SqueezeNet (torchvision)
def squeezenet_recipe():
    import torchvision.models as models
    model = models.squeezenet1_1(weights=None)
    x = torch.randn(2, 3, 64, 64)
    targets = torch.randint(0, 1000, (2,))
    return {
        "model": model,
        "sample_args": (x,),
        "loss_fn": lambda out: F.cross_entropy(out, targets),
    }


# 56. ShuffleNetV2 (torchvision)
def shufflenet_recipe():
    import torchvision.models as models
    model = models.shufflenet_v2_x0_5(weights=None)
    x = torch.randn(2, 3, 32, 32)
    targets = torch.randint(0, 1000, (2,))
    return {
        "model": model,
        "sample_args": (x,),
        "loss_fn": lambda out: F.cross_entropy(out, targets),
    }


# 57. ConvNeXt-Tiny (torchvision)
def convnext_recipe():
    import torchvision.models as models
    model = models.convnext_tiny(weights=None)
    x = torch.randn(2, 3, 32, 32)
    targets = torch.randint(0, 1000, (2,))
    return {
        "model": model,
        "sample_args": (x,),
        "loss_fn": lambda out: F.cross_entropy(out, targets),
    }


# 58. Swin Transformer Tiny (torchvision)
# NOTE: Swin uses aten.set_.source_Storage for shifted window attention,
# which fails with FakeTensors (meta/cpu device mismatch). Known PyTorch limitation.
# Kept as recipe but excluded from default wave4 runs.
def swin_recipe():
    import torchvision.models as models
    model = models.swin_t(weights=None)
    # Swin needs input size divisible by window_size * patch_size (7*4=28 minimum)
    x = torch.randn(2, 3, 56, 56)
    targets = torch.randint(0, 1000, (2,))
    return {
        "model": model,
        "sample_args": (x,),
        "loss_fn": lambda out: F.cross_entropy(out, targets),
    }


# 59. VGG-11 (torchvision) - classic deep CNN
def vgg11_recipe():
    import torchvision.models as models
    model = models.vgg11(weights=None)
    x = torch.randn(1, 3, 32, 32)  # small batch due to memory
    targets = torch.randint(0, 1000, (1,))
    return {
        "model": model,
        "sample_args": (x,),
        "loss_fn": lambda out: F.cross_entropy(out, targets),
    }


# 60. Vision Transformer (ViT-B/32, torchvision)
def vit_torchvision_recipe():
    import torchvision.models as models
    model = models.vit_b_32(weights=None)
    # ViT-B/32 needs 224x224 minimum
    x = torch.randn(1, 3, 224, 224)
    targets = torch.randint(0, 1000, (1,))
    return {
        "model": model,
        "sample_args": (x,),
        "loss_fn": lambda out: F.cross_entropy(out, targets),
    }


# 61. RegNet (torchvision)
def regnet_recipe():
    import torchvision.models as models
    model = models.regnet_x_400mf(weights=None)
    x = torch.randn(2, 3, 32, 32)
    targets = torch.randint(0, 1000, (2,))
    return {
        "model": model,
        "sample_args": (x,),
        "loss_fn": lambda out: F.cross_entropy(out, targets),
    }


# 62. MNASNet (torchvision)
def mnasnet_recipe():
    import torchvision.models as models
    model = models.mnasnet0_5(weights=None)
    x = torch.randn(2, 3, 32, 32)
    targets = torch.randint(0, 1000, (2,))
    return {
        "model": model,
        "sample_args": (x,),
        "loss_fn": lambda out: F.cross_entropy(out, targets),
    }


# 63. Wide ResNet-50 (torchvision)
def wide_resnet_recipe():
    import torchvision.models as models
    model = models.wide_resnet50_2(weights=None)
    # Need batch>1 since deep downsampling produces 1x1 spatial, and BN
    # requires >1 values per channel in training mode
    x = torch.randn(2, 3, 32, 32)
    targets = torch.randint(0, 1000, (2,))
    return {
        "model": model,
        "sample_args": (x,),
        "loss_fn": lambda out: F.cross_entropy(out, targets),
    }


# ═══════════════════════════════════════════════════════════════════════
# Wave 5: HuggingFace Transformers models
# ═══════════════════════════════════════════════════════════════════════

# 64. HuggingFace BERT (encoder-only)
def hf_bert_recipe():
    from transformers import BertConfig, BertModel
    config = BertConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                        intermediate_size=128, vocab_size=256, max_position_embeddings=64)
    model = BertModel(config)
    x = torch.randint(0, 256, (2, 32))
    return {
        "model": model,
        "sample_args": (x,),
        "loss_fn": lambda out: out.last_hidden_state.sum(),
    }


# 65. HuggingFace DistilBERT
def hf_distilbert_recipe():
    from transformers import DistilBertConfig, DistilBertModel
    config = DistilBertConfig(dim=64, n_layers=2, n_heads=2, hidden_dim=128,
                               vocab_size=256, max_position_embeddings=64)
    model = DistilBertModel(config)
    x = torch.randint(0, 256, (2, 32))
    return {
        "model": model,
        "sample_args": (x,),
        "loss_fn": lambda out: out.last_hidden_state.sum(),
    }


# 66. HuggingFace RoBERTa
def hf_roberta_recipe():
    from transformers import RobertaConfig, RobertaModel
    config = RobertaConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                            intermediate_size=128, vocab_size=256, max_position_embeddings=66)
    model = RobertaModel(config)
    x = torch.randint(1, 256, (2, 32))
    return {
        "model": model,
        "sample_args": (x,),
        "loss_fn": lambda out: out.last_hidden_state.sum(),
    }


# 67. SimpleViT (lucidrains/vit-pytorch)
def simple_vit_recipe():
    from vit_pytorch import SimpleViT
    model = SimpleViT(image_size=32, patch_size=8, num_classes=10, dim=64,
                      depth=2, heads=4, mlp_dim=128)
    x = torch.randn(2, 3, 32, 32)
    targets = torch.randint(0, 10, (2,))
    return {
        "model": model,
        "sample_args": (x,),
        "loss_fn": lambda out: F.cross_entropy(out, targets),
    }


# 68. Mixture of Experts (custom top-k gating)
def moe_topk_recipe():
    class Expert(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        def forward(self, x):
            return self.net(x)

    class MoE(nn.Module):
        def __init__(self, dim=64, n_experts=4, top_k=2):
            super().__init__()
            self.gate = nn.Linear(dim, n_experts)
            self.experts = nn.ModuleList([Expert(dim) for _ in range(n_experts)])
            self.top_k = top_k

        def forward(self, x):
            # x: (B, T, D)
            gate_logits = self.gate(x)  # (B, T, n_experts)
            weights, indices = gate_logits.topk(self.top_k, dim=-1)  # (B, T, k)
            weights = F.softmax(weights, dim=-1)
            # Simple: compute all experts, weight by gate
            expert_outs = torch.stack([e(x) for e in self.experts], dim=-2)  # (B, T, E, D)
            # Gather top-k expert outputs
            indices_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, x.size(-1))
            selected = expert_outs.gather(2, indices_expanded)  # (B, T, k, D)
            return (selected * weights.unsqueeze(-1)).sum(dim=2)  # (B, T, D)

    model = nn.Sequential(
        nn.Linear(32, 64),
        MoE(dim=64, n_experts=4, top_k=2),
        nn.Linear(64, 10),
    )
    x = torch.randn(2, 16, 32)
    targets = torch.randint(0, 10, (2, 16))
    return {
        "model": model,
        "sample_args": (x,),
        "loss_fn": lambda out: F.cross_entropy(out.view(-1, 10), targets.view(-1)),
    }


# 69. Perceiver-style cross-attention
def perceiver_recipe():
    class CrossAttention(nn.Module):
        def __init__(self, dim=64, heads=4):
            super().__init__()
            self.to_q = nn.Linear(dim, dim)
            self.to_kv = nn.Linear(dim, dim * 2)
            self.to_out = nn.Linear(dim, dim)
            self.heads = heads
            self.scale = (dim // heads) ** -0.5

        def forward(self, latents, data):
            B, N, D = latents.shape
            h = self.heads
            q = self.to_q(latents).view(B, N, h, D // h).transpose(1, 2)
            kv = self.to_kv(data).view(B, -1, 2, h, D // h).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            out = (attn @ v).transpose(1, 2).reshape(B, N, D)
            return self.to_out(out)

    class MiniPerceiver(nn.Module):
        def __init__(self, dim=64, n_latents=8, depth=2):
            super().__init__()
            self.latents = nn.Parameter(torch.randn(1, n_latents, dim))
            self.cross_attns = nn.ModuleList([CrossAttention(dim) for _ in range(depth)])
            self.head = nn.Linear(dim, 10)

        def forward(self, data):
            latents = self.latents.expand(data.size(0), -1, -1)
            for ca in self.cross_attns:
                latents = latents + ca(latents, data)
            return self.head(latents.mean(dim=1))

    model = MiniPerceiver()
    x = torch.randn(2, 32, 64)  # data tokens
    targets = torch.randint(0, 10, (2,))
    return {
        "model": model,
        "sample_args": (x,),
        "loss_fn": lambda out: F.cross_entropy(out, targets),
    }


# 70. Causal Transformer (custom GPT-like, no KV cache issues)
def causal_transformer_recipe():
    class CausalTransformer(nn.Module):
        def __init__(self, vocab_size=256, dim=64, n_heads=4, n_layers=2, seq_len=32):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, dim)
            self.pos_emb = nn.Embedding(seq_len, dim)
            layer = nn.TransformerEncoderLayer(dim, n_heads, dim * 4, dropout=0.0, batch_first=True)
            self.blocks = nn.TransformerEncoder(layer, n_layers)
            self.head = nn.Linear(dim, vocab_size)
            self.register_buffer('causal_mask',
                torch.nn.Transformer.generate_square_subsequent_mask(seq_len))

        def forward(self, x):
            B, T = x.shape
            pos = torch.arange(T, device=x.device)
            h = self.tok_emb(x) + self.pos_emb(pos)
            mask = self.causal_mask[:T, :T]
            h = self.blocks(h, mask=mask, is_causal=True)
            return self.head(h)

    model = CausalTransformer()
    x = torch.randint(0, 256, (2, 32))
    targets = torch.randint(0, 256, (2, 32))
    return {
        "model": model,
        "sample_args": (x,),
        "loss_fn": lambda out: F.cross_entropy(out.view(-1, 256), targets.view(-1)),
    }


# 71. DenseNet-121 (torchvision)
def densenet121_recipe():
    import torchvision.models as models
    model = models.densenet121(weights=None)
    x = torch.randn(2, 3, 32, 32)
    targets = torch.randint(0, 1000, (2,))
    return {
        "model": model,
        "sample_args": (x,),
        "loss_fn": lambda out: F.cross_entropy(out, targets),
    }


# 72. AlexNet (torchvision) - classic
def alexnet_recipe():
    import torchvision.models as models
    model = models.alexnet(weights=None)
    x = torch.randn(2, 3, 64, 64)
    targets = torch.randint(0, 1000, (2,))
    return {
        "model": model,
        "sample_args": (x,),
        "loss_fn": lambda out: F.cross_entropy(out, targets),
    }


def hf_gpt2_recipe():
    from transformers import GPT2LMHeadModel, GPT2Config
    config = GPT2Config(n_layer=2, n_head=2, n_embd=128, vocab_size=1000)
    model = GPT2LMHeadModel(config)
    model.train()
    x = torch.randint(0, 1000, (1, 16))
    labels = torch.randint(0, 1000, (1, 16))
    return {
        "model": model,
        "sample_args": (x,),
        "sample_kwargs": {"labels": labels},
    }


WAVE5_RECIPES = {
    "hf_bert": hf_bert_recipe,
    "hf_distilbert": hf_distilbert_recipe,
    "hf_roberta": hf_roberta_recipe,
    "hf_gpt2": hf_gpt2_recipe,
    "simple_vit": simple_vit_recipe,
    "moe_topk": moe_topk_recipe,
    "perceiver": perceiver_recipe,
    "causal_transformer": causal_transformer_recipe,
    "densenet121": densenet121_recipe,
    "alexnet": alexnet_recipe,
}

WAVE4_RECIPES = {
    "resnet18": resnet18_recipe,
    "mobilenetv2": mobilenetv2_recipe,
    "efficientnet_b0": efficientnet_b0_recipe,
    "squeezenet": squeezenet_recipe,
    "shufflenet": shufflenet_recipe,
    "convnext": convnext_recipe,
    # swin: excluded - aten.set_.source_Storage FakeTensor limitation
    "vgg11": vgg11_recipe,
    "vit_torchvision": vit_torchvision_recipe,
    "regnet": regnet_recipe,
    "mnasnet": mnasnet_recipe,
    "wide_resnet": wide_resnet_recipe,
}

WAVE3_RECIPES = {
    "mingpt": mingpt_recipe,
    "fn_mingpt_block": fn_mingpt_block,
    "vit": vit_recipe,
    "fn_vit_attention": fn_vit_attention,
    "pytorch_mnist": pytorch_mnist_recipe,
    "pytorch_vae": pytorch_vae_recipe,
    "pytorch_wlm": pytorch_wlm_recipe,
    "pytorch_superres": pytorch_superres_recipe,
    "pytorch_dcgan_gen": pytorch_dcgan_gen_recipe,
    "pytorch_regression": pytorch_regression_recipe,
    "nanochat": nanochat_recipe,
    "nanochat_sft": nanochat_sft_recipe,
    "nanochat_rl": nanochat_rl_recipe,
}

WAVE2_RECIPES = {
    "densenet": densenet_recipe,
    "distillation": distillation_recipe,
    "swiglu": swiglu_recipe,
    "gnn": gnn_recipe,
    "batch_renorm": batch_renorm_recipe,
    "multitask": multitask_recipe,
    "grid_sample": grid_sample_recipe,
    "mdn": mdn_recipe,
    "cond_bn": cond_bn_recipe,
    "spectral_norm": spectral_norm_recipe,
}


def run_one(name, recipe_fn, output_base="outputs/test_models", is_fn_recipe=False,
            device="cuda", verify_pool=None):
    """Run extraction on one recipe. Returns (success, error_msg, elapsed_secs)."""
    import time as _time
    import glob as _glob
    _t0 = _time.perf_counter()
    output_dir = os.path.join(output_base, name)

    print(f"\n{'='*60}")
    print(f" [{name}]{'  (function)' if is_fn_recipe else ''}")
    print(f"{'='*60}")

    # Save sys.path so recipes that insert paths don't pollute later runs
    saved_path = sys.path[:]
    try:
        recipe = recipe_fn()

        # Move any tensors captured in the loss_fn closure to the target device
        loss_fn = recipe.get("loss_fn")
        if loss_fn is not None and device != "cpu" and hasattr(loss_fn, '__closure__') and loss_fn.__closure__:
            for cell in loss_fn.__closure__:
                obj = cell.cell_contents
                if isinstance(obj, torch.Tensor) and not obj.is_cuda:
                    cell.cell_contents = obj.to(device)

        if is_fn_recipe or "fn" in recipe:
            # Function recipe mode
            fn = recipe["fn"]
            fn_args = recipe.get("args", ())
            fn_kwargs = recipe.get("kwargs", {})
            loss_fn = recipe.get("loss_fn")
            run_backward = recipe.get("run_backward", loss_fn is not None)
            param_names = recipe.get("param_names")

            # Move any tensors captured in the fn closure to the target device
            if device != "cpu" and hasattr(fn, '__closure__') and fn.__closure__:
                for cell in fn.__closure__:
                    try:
                        obj = cell.cell_contents
                    except ValueError:
                        continue
                    if isinstance(obj, torch.Tensor) and not obj.is_cuda:
                        cell.cell_contents = obj.to(device)

            if isinstance(fn, nn.Module):
                n_params = sum(p.numel() for p in fn.parameters())
                print(f"  Params: {n_params:,}")
            else:
                print(f"  Function: {fn.__name__ if hasattr(fn, '__name__') else repr(fn)}")

            result = extract_function(
                fn, *fn_args,
                run_backward=run_backward,
                loss_fn=loss_fn,
                output_dir=output_dir,
                prefix=name,
                device=device,
                record_real_tensors=True,
                max_intermediates_mb=0,
                param_names=param_names,
                **fn_kwargs,
            )
        else:
            # Standard model recipe mode
            model = recipe["model"]
            sample_args = recipe["sample_args"]
            loss_fn = recipe.get("loss_fn")

            n_params = sum(p.numel() for p in model.parameters())
            print(f"  Params: {n_params:,}")

            result = extract_training_step(
                model=model,
                sample_args=sample_args,
                sample_kwargs=recipe.get("sample_kwargs"),
                loss_fn=loss_fn,
                output_dir=output_dir,
                prefix=name,
                device=device,
                record_real_tensors=True,
                max_intermediates_mb=0,
            )

        # Validate outputs
        files = result["files"]
        py_file = next((f for f in files if f.endswith("_aten.py")), None)
        html_file = next((f for f in files if f.endswith(".html")), None)

        if not py_file:
            return False, "No _aten.py generated"
        if not html_file:
            return False, "No .html generated"
        if not os.path.exists(py_file):
            return False, f"_aten.py not on disk: {py_file}"
        if not os.path.exists(html_file):
            return False, f".html not on disk: {html_file}"

        # Try to verify the exported script
        print(f"  Verifying exported script …")
        if verify_pool:
            ok, err = verify_pool.verify(py_file, device=device)
            if not ok:
                return False, err, _time.perf_counter() - _t0
        else:
            import subprocess as _subprocess
            cmd = [sys.executable, py_file]
            if device == "cpu":
                cmd.append("--cpu")
            verify = _subprocess.run(
                cmd, capture_output=True, text=True, timeout=120,
            )
            if verify.returncode != 0:
                return False, f"Verify failed (rc={verify.returncode}):\n{verify.stderr[-2000:]}", _time.perf_counter() - _t0

        loss = result.get("loss_value")
        elapsed = _time.perf_counter() - _t0
        print(f"  OK  loss={loss:.4f}  ({elapsed:.1f}s)" if loss else f"  OK  ({elapsed:.1f}s)")
        return True, None, elapsed

    except Exception as e:
        tb = traceback.format_exc()
        return False, f"{e}\n{tb[-800:]}", _time.perf_counter() - _t0
    finally:
        sys.path[:] = saved_path
        # Clean up large .pt files to avoid filling disk
        for pt_file in _glob.glob(os.path.join(output_dir, "*.pt")):
            try:
                os.remove(pt_file)
            except OSError:
                pass


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated list of model names to test")
    parser.add_argument("--skip", type=str, default=None,
                        help="Comma-separated list to skip")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU (default: use CUDA if available)")
    args = parser.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    print(f"Device: {device}")

    # Combine all recipe sets
    all_recipes = dict(ALL_RECIPES)
    all_recipes.update(WAVE2_RECIPES)
    all_recipes.update(WAVE3_RECIPES)
    all_recipes.update(WAVE4_RECIPES)
    all_recipes.update(WAVE5_RECIPES)
    fn_recipe_names = set(FN_RECIPES.keys()) | {k for k in WAVE3_RECIPES if k.startswith("fn_")}
    for name, fn in FN_RECIPES.items():
        all_recipes[name] = fn

    recipes = all_recipes
    if args.only:
        names = [n.strip() for n in args.only.split(",")]
        recipes = {n: recipes[n] for n in names if n in recipes}
    if args.skip:
        skip = {n.strip() for n in args.skip.split(",")}
        recipes = {n: v for n, v in recipes.items() if n not in skip}

    output_base = "outputs/test_models"
    os.makedirs(output_base, exist_ok=True)

    # Pre-warm a pool of Python subprocesses with torch already imported.
    # Each test takes ~2-3s for extraction; worker warmup takes ~1.5s.
    # With n_warm=min(8, n_tests), workers are ready well before needed.
    n_tests = len(recipes)
    verify_pool = VerifyPool(n_warm=min(8, n_tests))

    results = {}
    for name, fn in recipes.items():
        ok, err, elapsed = run_one(name, fn, output_base, is_fn_recipe=(name in fn_recipe_names),
                                   device=device, verify_pool=verify_pool)
        results[name] = (ok, err, elapsed)
        # Reset torch.compile caches between models
        torch.compiler.reset()

    verify_pool.shutdown()

    # Summary
    print("\n")
    print("=" * 60)
    print(" SUMMARY")
    print("=" * 60)
    passed = sum(1 for ok, _, _ in results.values() if ok)
    failed = sum(1 for ok, _, _ in results.values() if not ok)
    total = len(results)
    total_time = sum(t for _, _, t in results.values())

    for name, (ok, err, elapsed) in sorted(results.items(), key=lambda x: -x[1][2]):
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name:30s} {elapsed:6.1f}s")
        if err:
            # Show first line of error
            first_line = err.split("\n")[0]
            print(f"         {first_line}")

    print(f"\n  {passed}/{total} passed, {failed} failed, total {total_time:.1f}s")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
