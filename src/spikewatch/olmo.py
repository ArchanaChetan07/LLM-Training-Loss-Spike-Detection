"""OLMo W&B telemetry -> one monitoring stream per training series.

A *series* is the sequence of W&B runs that make up one model's pre-training
(OLMo-1B-run-001 ... run-063). Runs are restarts; a restart after a spike
usually rewinds to an earlier checkpoint, so step numbers overlap. We keep
every run's full trajectory in wall-clock order: that is exactly what an
online monitor would have seen, including the spike that caused the rewind.

Derived signals (layer stats are logged every 10 steps; loss/lr/clip every step):

  loss    train cross-entropy                                     every step
  clip    fraction of grads clipped (OLMo `optim/clipping_rate`)  every step
  gnorm   ||g||, total over matrix weights                        10 steps (1 step if logged)
  gmax    max_i ||g_i|| / median of its own last 50 logs          10 steps
          (spikes are often localised: OLMo-1B's start in block 0 att_proj)
  uratio  Adam update-to-weight ratio, lr ||m / sqrt(v)|| / ||theta||,
          with ||m_i / sqrt(v_i)|| ~ ||m_i|| / sqrt(mean v_i) per tensor
  stale   Adam second-moment staleness, sum_i ||g_i||^2 / sum_i n_i mean(v_i):
          ~1 at equilibrium, >>1 when the gradient outruns v and the
          effective step blows up (Molybog et al. 2023)
  curv    secant curvature proxy, d log||g|| per unit relative displacement
          between consecutive logs: how fast the gradient reacts to movement
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

SERIES = {
    "OLMo-1B": ("OLMo-1B", r"^OLMo-1B-run-\d+$"),
    "OLMo-7B": ("OLMo-7B", r"^OLMo-7B-run-\d+$"),
    "OLMo-1.7-7B": ("OLMo-7B", r"^OLMo-1\.7-7B-run-\d+$"),
    "OLMo-2-7B": ("OLMo-2-1124-7B", r"^OLMo-2-1124-7B-stage-1-run-\d+$"),
}

_TENSOR = re.compile(r"transformer\.(?:blocks\.\d+\.)?([a-z_]+)\.weight")


def _numel(name: str, m: dict) -> int:
    d = m["d_model"]
    hidden = m.get("mlp_hidden_size") or m["mlp_ratio"] * d
    kind = _TENSOR.search(name).group(1)
    heads = m["n_heads"]
    kv = m.get("n_kv_heads") or (1 if m.get("multi_query_attention") else heads)
    if kind == "att_proj":
        return (d + 2 * kv * d // heads) * d
    if kind == "attn_out":
        return d * d
    if kind == "ff_proj":
        return hidden * d
    if kind == "ff_out" and ".blocks." in name:
        return d * hidden // 2  # SwiGLU halves the projection
    if kind in ("wte", "ff_out"):
        return (m.get("embedding_size") or m["vocab_size"]) * d
    raise KeyError(name)


def run_dirs(series: str) -> list[Path]:
    proj, pat = SERIES[series]
    rx = re.compile(pat)
    base = RAW / proj
    ds = [d for d in base.iterdir() if d.is_dir() and rx.match(d.name) and (d / "base.parquet").exists()]
    return sorted(ds, key=lambda d: int(re.search(r"(\d+)$", d.name).group(1)))


def _layer_features(lay: pd.DataFrame, lr: pd.Series, cfg: dict) -> pd.DataFrame:
    cols = lay.columns
    g = [c for c in cols if c.startswith("optim/grad/")]
    key = lambda c: c.split("/", 2)[2].rsplit(".", 1)[0]  # tensor name
    m = {key(c): c for c in cols if c.startswith("optim/exp_avg/")}
    v = {key(c): c for c in cols if c.startswith("optim/exp_avg_sq/")}
    p = {key(c): c for c in cols if c.startswith("optim/param/")}
    names = [key(c) for c in g if key(c) in m and key(c) in v and key(c) in p]
    G2 = sum(lay[f"optim/grad/{n}.norm"].astype(float) ** 2 for n in names)
    P2 = sum(lay[p[n]].astype(float) ** 2 for n in names)
    MV = sum(lay[m[n]].astype(float) ** 2 / np.maximum(lay[v[n]].astype(float), 1e-30) for n in names)
    model = cfg.get("model") or {}
    try:
        NV = sum(_numel(n, model) * lay[v[n]].astype(float) for n in names)
    except (KeyError, TypeError, AttributeError):
        NV = np.nan
    out = pd.DataFrame({"step": lay["_step"].to_numpy()})
    out["gnorm_l"] = np.sqrt(G2.to_numpy())
    lr_at = lr.reindex(out["step"]).to_numpy()
    out["uratio"] = lr_at * np.sqrt(MV.to_numpy()) / np.sqrt(P2.to_numpy())
    out["stale"] = (G2 / NV).to_numpy() if not np.isscalar(NV) else np.nan
    gi = lay[[f"optim/grad/{n}.norm" for n in names]].astype(float)
    ratio = gi / gi.rolling(50, min_periods=10).median().shift(1)
    out["gmax"] = ratio.max(axis=1, skipna=True).to_numpy()
    # secant curvature proxy between consecutive logs
    dlogg = np.diff(np.log(out["gnorm_l"].to_numpy()), prepend=np.nan)
    disp = out["uratio"].rolling(2).sum().to_numpy() * np.maximum(np.diff(out["step"], prepend=np.nan), 1) / 2
    out["curv"] = dlogg / np.maximum(disp, 1e-12)
    return out


def load_series(series: str, cache: bool = True) -> pd.DataFrame:
    """One row per logged optimizer step, wall-clock order.

    Columns: run, step, pos (0..N-1), seg_start, loss, lr, clip,
             gnorm, gmax, uratio, stale, curv (NaN off the 10-step cadence).
    """
    cpath = ROOT / "data" / "processed" / f"{series}.parquet"
    if cache and cpath.exists():
        return pd.read_parquet(cpath)
    parts = []
    for i, d in enumerate(run_dirs(series)):
        b = pd.read_parquet(d / "base.parquet")
        if b.empty or "_step" not in b or "train/CrossEntropyLoss" not in b:
            continue  # empty / crashed run
        b = b.drop_duplicates("_step", keep="last").sort_values("_step")
        b = b.rename(columns={"_step": "step", "train/CrossEntropyLoss": "loss",
                              "optim/learning_rate_group0": "lr", "optim/clipping_rate": "clip",
                              "optim/total_grad_norm": "gnorm_s"})
        b = b.dropna(subset=["loss"])
        cfg = json.loads((d / "config.json").read_text()) if (d / "config.json").exists() else {}
        lp = d / "layers.parquet"
        if lp.exists():
            lay = pd.read_parquet(lp).drop_duplicates("_step", keep="last").sort_values("_step")
            if len(lay):
                lf = _layer_features(lay, b.set_index("step")["lr"], cfg)
                b = b.merge(lf, on="step", how="left")
        b.insert(0, "run", d.name)
        b["run_idx"] = i
        parts.append(b)
    df = pd.concat(parts, ignore_index=True)
    for c in ("clip", "gnorm_s", "gnorm_l", "uratio", "stale", "gmax", "curv", "lr"):
        if c not in df:
            df[c] = np.nan
    # a per-step total grad norm (OLMo-2) beats the 10-step layer aggregate
    df["gnorm"] = df["gnorm_s"].where(df["gnorm_s"].notna(), df["gnorm_l"])
    df["pos"] = np.arange(len(df))
    df["seg_start"] = df["run"].ne(df["run"].shift())
    keep = ["run", "run_idx", "step", "pos", "seg_start", "loss", "lr", "clip", "gnorm", "gmax",
            "uratio", "stale", "curv"]
    df = df[keep]
    if cache:
        cpath.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cpath)
    return df
