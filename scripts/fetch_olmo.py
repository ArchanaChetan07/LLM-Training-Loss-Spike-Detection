"""Fetch OLMo training telemetry from AI2's public W&B projects.

Two tables per W&B run, cached as parquet under data/raw/<project>/<run>/:
  base.parquet   every optimizer step: loss, lr, clipping rate (+ total grad
                 norm / z-loss where logged)
  layers.parquet every logged step (OLMo logs per-tensor stats every 10 steps):
                 per-tensor grad / param / exp_avg L2 norms and exp_avg_sq mean

Idempotent: a run whose files exist is skipped, so the script can be killed
and resumed. Usage:
    python scripts/fetch_olmo.py ai2-llm/OLMo-1B [--prefix OLMo-1B-run]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE_KEYS = [
    "_step",
    "train/CrossEntropyLoss",
    "train/ZLoss",
    "optim/learning_rate_group0",
    "optim/clipping_rate",
    "optim/total_grad_norm",
]
LAYER_RE = re.compile(
    r"^optim/(grad|exp_avg|param)/transformer\..*weight\.norm$"
    r"|^optim/exp_avg_sq/transformer\..*weight\.avg$"
)
# OLMo-2 logs q_norm/k_norm/attn_norm/ff_norm too; their grads are tiny and
# they triple the key count. Matrix weights carry the signal; keep those.
SKIP_RE = re.compile(r"(q_norm|k_norm|attn_norm|ff_norm|ln_f)\.weight")


def scan(run, keys, page_size):
    """scan_history only returns rows where *all* requested keys are present.
    Keys logged at different cadences (z-loss every 10 steps, loss every step)
    would silently thin the table, so callers pass keys of one cadence only."""
    present = set(run.summary.keys())
    keys = [k for k in keys if k == "_step" or k in present]
    rows = list(run.scan_history(keys=keys, page_size=page_size))
    return pd.DataFrame(rows, columns=keys), keys


def scan_base(run):
    """Fetch each per-step key on its own and outer-join on _step."""
    df = None
    for k in BASE_KEYS[1:]:
        part, got = scan(run, ["_step", k], 10000)
        if len(got) < 2 or part.empty:
            continue
        df = part if df is None else df.merge(part, on="_step", how="outer")
    return (df if df is not None else pd.DataFrame(columns=["_step"])).sort_values("_step")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("--prefix", default="", help="only runs whose name starts with this")
    ap.add_argument("--no-layers", action="store_true")
    args = ap.parse_args()

    import wandb

    api = wandb.Api(timeout=600)
    out_root = ROOT / "data" / "raw" / args.project.split("/")[-1]
    out_root.mkdir(parents=True, exist_ok=True)
    runs = [r for r in api.runs(args.project, per_page=200) if r.name.startswith(args.prefix)]
    print(f"{args.project}: {len(runs)} runs", flush=True)
    index = []
    for r in sorted(runs, key=lambda r: r.name):
        d = out_root / r.name
        meta = {"name": r.name, "id": r.id, "state": r.state, "created_at": r.created_at}
        index.append(meta)
        if (d / "base.parquet").exists() and (args.no_layers or (d / "layers.parquet").exists()):
            continue
        d.mkdir(exist_ok=True)
        t0 = time.time()
        for attempt in range(4):
            try:
                base = scan_base(r)
                base.to_parquet(d / "base.parquet")
                if not args.no_layers:
                    lk = [k for k in r.summary.keys() if LAYER_RE.match(k) and not SKIP_RE.search(k)]
                    lay, _ = scan(r, ["_step"] + sorted(lk), 1000)
                    lay.astype({c: "float32" for c in lay.columns if c != "_step"}).to_parquet(
                        d / "layers.parquet"
                    )
                cfg = {k: r.config.get(k) for k in ("model", "optimizer", "scheduler", "max_grad_norm",
                                                      "max_grad_norm_ratio", "global_train_batch_size")}
                (d / "config.json").write_text(json.dumps(cfg, default=str, indent=1))
                break
            except Exception as e:  # network hiccups on long scans
                print(f"  retry {r.name}: {type(e).__name__}: {str(e)[:120]}", flush=True)
                time.sleep(10 * (attempt + 1))
        else:
            print(f"  FAILED {r.name}", flush=True)
            continue
        n = len(base)
        print(f"  {r.name}: {n} steps [{base['_step'].min() if n else '-'}..{base['_step'].max() if n else '-'}]"
              f" {time.time()-t0:.0f}s", flush=True)
    (out_root / "index.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    sys.exit(main())
