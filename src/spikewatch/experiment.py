"""Protocol in docs/PREREGISTRATION.md, as code."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .audit import fit_stateful
from .baselines import cusum_alarms, zscore_alarms
from .evalues import StatefulCalibrator, TailCalibrator
from .labels import label_tiers
from .metrics import evaluate
from .pipeline import Scored, detect, recalibrate, score

SIGNALS = ["loss", "clip", "gnorm", "gmax", "uratio", "stale", "curv"]
OPTIM_ONLY = ["gnorm", "gmax", "uratio", "stale", "curv"]
H, GRACE, W, ALPHA_W, ALPHA_RUN = 200, 200, 1000, 0.05, 0.05
Z_GRID = [4, 5, 6, 7, 8, 10, 12, 15, 20, 30]
CUSUM_GRID = [5, 10, 20, 40, 80, 160]


@dataclass
class Prepared:
    name: str
    df: pd.DataFrame
    major: list
    minor: list
    sc: Scored          # p-values (+ rho=1 e-values)
    armed: np.ndarray

    def mask_stable(self) -> np.ndarray:
        n = len(self.df)
        ex = np.zeros(n, bool)
        for e in list(self.major) + list(self.minor):
            ex[max(0, e.onset - H): min(n, e.end + GRACE + 1)] = True
        return self.armed & ~ex


def prepare(name: str, df: pd.DataFrame, burn_in: int, signals=SIGNALS, specs=None, seed: int = 0) -> Prepared:
    major, minor = label_tiers(df, burn_in=burn_in)
    sc = score(df, signals, TailCalibrator(), burn_in=burn_in, specs=specs, seed=seed)
    return Prepared(name, df, major, minor, sc, sc.armed)


def sub(p: Prepared, m: np.ndarray):
    """Episodes and armed mask restricted to positions where m is True."""
    return ([e for e in p.major if m[e.onset]], [e for e in p.minor if m[e.onset]], p.armed & m)


def run_detector(p: Prepared, cal, threshold: float, signals=None, m: np.ndarray | None = None) -> dict:
    sc = recalibrate(p.sc, cal) if cal is not None else p.sc
    logR, alarms, e = detect(sc, threshold, signals=signals)
    m = np.ones(len(p.df), bool) if m is None else m
    major, minor, armed = sub(p, m)
    r = evaluate(alarms, major, len(p.df), armed=armed, horizon=H, grace=GRACE, window=W, ignore=minor)
    r_strict = evaluate(alarms, major, len(p.df), armed=armed, horizon=H, grace=GRACE, window=W)
    out = r.summary()
    out["window_fa_rate_strict"] = r_strict.window_fa_rate
    out["lead_onset"] = r.lead_onset.tolist()
    out["lead_peak"] = r.lead_peak.tolist()
    out["_windows"], out["_wfa"] = r.windows, r.windows_with_fa
    out["windows"], out["windows_with_fa"] = int(r.windows), int(r.windows_with_fa)
    out["_alarms"] = alarms
    out["_logR"] = logR
    return out


def baseline_eval(p: Prepared, kind: str, param: float, m=None, signals=SIGNALS) -> dict:
    """Tuned classical detector on the same causal robust z-scores."""
    z = {s: p.sc.resid[s] for s in signals if s in p.sc.resid}
    z = {s: np.where(p.armed, v, np.nan) for s, v in z.items()}
    al = zscore_alarms(z, param) if kind == "z" else cusum_alarms(z, param)
    m = np.ones(len(p.df), bool) if m is None else m
    major, minor, armed = sub(p, m)
    r = evaluate(al, major, len(p.df), armed=armed, horizon=H, grace=GRACE, window=W, ignore=minor)
    out = r.summary(); out["param"] = param
    out["lead_onset"] = r.lead_onset.tolist(); out["lead_peak"] = r.lead_peak.tolist()
    out["_windows"], out["_wfa"] = r.windows, r.windows_with_fa
    out["windows"], out["windows_with_fa"] = int(r.windows), int(r.windows_with_fa)
    return out


POOL_KEYS = ["events", "detected", "detected_ahead_of_onset", "detected_before_peak", "false_alarms",
             "stable_steps"]


def pool(rs: list[dict]) -> dict:
    """Pool per-run results: counts add, leads concatenate, window FA is exact."""
    out = {k: int(sum(r[k] for r in rs)) for k in POOL_KEYS}
    fin = lambda xs: [float(x) for x in xs if x is not None and np.isfinite(x)]
    lo = fin(x for r in rs for x in r["lead_onset"]); lp = fin(x for r in rs for x in r["lead_peak"])
    out["median_lead_onset"] = float(np.median(lo)) if lo else float("nan")
    out["median_lead_peak"] = float(np.median(lp)) if lp else float("nan")
    out["lead_onset"] = lo
    out["windows"] = int(sum(r["_windows"] for r in rs)); out["windows_with_fa"] = int(sum(r["_wfa"] for r in rs))
    out["window_fa_rate"] = out["windows_with_fa"] / out["windows"] if out["windows"] else float("nan")
    out["fa_per_10k"] = 1e4 * out["false_alarms"] / max(out["stable_steps"], 1)
    return out


def tune_baseline(p: Prepared, kind: str, m: np.ndarray, target: float = ALPHA_W) -> float:
    grid = Z_GRID if kind == "z" else CUSUM_GRID
    best = grid[-1]
    for g in sorted(grid, reverse=True):  # from conservative to sensitive
        if baseline_eval(p, kind, g, m)["window_fa_rate"] <= target:
            best = g
        else:
            break
    return best


def cal_to_json(cals: dict[str, StatefulCalibrator]) -> dict:
    return {s: {"betas": c.betas, "rho0": c.rho0, "rho1": c.rho1, "K": c.K} for s, c in cals.items()}


def cal_from_json(d: dict) -> dict[str, StatefulCalibrator]:
    return {s: StatefulCalibrator(tuple(v["betas"]), tuple(v["rho0"]), tuple(v["rho1"]), v["K"]) for s, v in d.items()}


def clean(d):
    """Strip private arrays and make JSON-safe."""
    if isinstance(d, dict):
        return {k: clean(v) for k, v in d.items() if not k.startswith("_")}
    if isinstance(d, (list, tuple)):
        return [clean(v) for v in d]
    if isinstance(d, (np.floating, float)):
        return None if not np.isfinite(d) else float(d)
    if isinstance(d, np.integer):
        return int(d)
    return d
