"""Small GPT runs with full per-step instability telemetry.

Why: OLMo logs optimizer stats every 10 steps and only as per-tensor norms, so
update ratio and curvature have to be approximated. Here every quantity is
exact and logged every step:

  loss     train cross-entropy (bytes)
  gnorm    ||g|| before clipping
  gmax     max over tensors of ||g_i|| / its running median
  uratio   ||theta_t - theta_{t-1}|| / ||theta_{t-1}||          (exact update ratio)
  stale    sum g^2 / sum v   with v Adam's exp_avg_sq           (second-moment staleness)
  curv     (g_t - g_{t-1}) . d_{t-1} / ||d_{t-1}||^2, d the last update
           -> secant (Barzilai-Borwein) curvature along the trajectory
  logitmax max |output logit| (logit growth, Wortsman et al. 2023)

Instability is induced the way it appears at scale: high learning rate, short
warmup, no QK-norm, no z-loss. The corpus is byte-level Python source taken
from the local interpreter's standard library and site-packages (no download).

  python scripts/train_small.py --lr 6e-3 --seed 0 --steps 6000
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus.bin"


def build_corpus(max_bytes: int = 120_000_000):
    if CORPUS.exists():
        return
    roots = [Path(os.__file__).parent]
    buf, total = [], 0
    for r in roots:
        for p in sorted(r.rglob("*.py")):
            try:
                b = p.read_bytes()
            except OSError:
                continue
            buf.append(b); total += len(b)
            if total >= max_bytes:
                break
    CORPUS.parent.mkdir(parents=True, exist_ok=True)
    CORPUS.write_bytes(b"\n".join(buf))
    print(f"corpus: {total/1e6:.1f} MB from {len(buf)} files", flush=True)


class Block(nn.Module):
    def __init__(self, d, h, post_ln=False):
        super().__init__()
        self.h = h
        self.post_ln = post_ln
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False); self.proj = nn.Linear(d, d, bias=False)
        self.fc = nn.Linear(d, 4 * d, bias=False); self.out = nn.Linear(4 * d, d, bias=False)

    def _attn(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, D // self.h).unbind(2)
        a = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        return self.proj(a.transpose(1, 2).reshape(B, T, D))

    def forward(self, x):
        if self.post_ln:  # original Transformer: LN after the residual add; unstable at high LR
            x = self.ln1(x + self._attn(x))
            return self.ln2(x + self.out(F.gelu(self.fc(x))))
        x = x + self._attn(self.ln1(x))
        return x + self.out(F.gelu(self.fc(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, d=256, L=6, h=8, T=256, V=256, post_ln=False):
        super().__init__()
        self.wte = nn.Embedding(V, d); self.wpe = nn.Embedding(T, d)
        self.blocks = nn.ModuleList(Block(d, h, post_ln) for _ in range(L))
        self.ln = nn.LayerNorm(d); self.head = nn.Linear(d, V, bias=False)
        for p in self.parameters():
            if p.dim() == 2:
                nn.init.normal_(p, 0, 0.02)

    def forward(self, idx):
        x = self.wte(idx) + self.wpe(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=6e-3)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--post-ln", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    build_corpus()
    data = np.memmap(CORPUS, dtype=np.uint8, mode="r")
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    dev = "cuda"
    model = GPT(a.d, a.layers, 8, a.seq, post_ln=a.post_ln).to(dev)
    params = [p for p in model.parameters()]
    decay = [p for p in params if p.dim() == 2]; nodecay = [p for p in params if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": nodecay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, a.beta2), eps=a.eps)
    sched = lambda s: min(1.0, (s + 1) / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
    name = a.out or f"lr{a.lr:g}_wu{a.warmup}_b2{a.beta2:g}_s{a.seed}"
    out = ROOT / "data" / "small" / f"{name}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)

    flat = lambda ts: torch.cat([t.reshape(-1) for t in ts])
    prev_g = prev_d = None
    gmed = None
    rows = []
    t0 = time.time()
    for step in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = a.lr * sched(step)
        ix = rng.integers(0, len(data) - a.seq - 1, a.batch)
        xb = torch.from_numpy(np.stack([data[i: i + a.seq + 1] for i in ix]).astype(np.int64)).to(dev)
        logits = model(xb[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, 256), xb[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            g_t = flat([p.grad for p in params])
            per = torch.stack([p.grad.norm() for p in params])
            gnorm = g_t.norm().item()
            gmed = per.clone() if gmed is None else 0.98 * gmed + 0.02 * per
            gmax = (per / gmed.clamp_min(1e-12)).max().item()
            curv = float("nan")
            if prev_d is not None:
                dd = (prev_d * prev_d).sum()
                if dd > 0:
                    curv = ((g_t - prev_g) * prev_d).sum().item() / dd.item()
            theta_prev = flat([p.detach() for p in params])
        torch.nn.utils.clip_grad_norm_(params, a.clip)
        opt.step()
        with torch.no_grad():
            theta = flat([p.detach() for p in params])
            d_t = theta - theta_prev
            uratio = (d_t.norm() / theta_prev.norm()).item()
            vs = [opt.state[p]["exp_avg_sq"] for p in params]
            stale = (g_t.pow(2).sum() / sum(v.sum() for v in vs).clamp_min(1e-30)).item()
            prev_g, prev_d = g_t, d_t
            logitmax = logits.detach().abs().amax().item()
        L = loss.item()
        rows.append((step, L, a.lr * sched(step), gnorm, gmax, uratio, stale, curv, logitmax,
                     float(gnorm > a.clip)))
        if not math.isfinite(L):
            print(f"{name}: non-finite loss at {step}", flush=True)
            break
        if step % 1000 == 0:
            print(f"{name} step {step} loss {L:.3f} gnorm {gnorm:.2f} {time.time()-t0:.0f}s", flush=True)
    import pandas as pd
    df = pd.DataFrame(rows, columns=["step", "loss", "lr", "gnorm", "gmax", "uratio", "stale", "curv",
                                     "logitmax", "clip"])
    df["run"] = name
    df.to_parquet(out)
    print(f"{name}: done {len(df)} steps in {time.time()-t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
