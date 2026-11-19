"""End-to-end: telemetry frame -> per-signal e-values -> alarms."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .detector import DEFAULT_LAMBDAS, combine, sr_statistic
from .evalues import SignalSpec, TailCalibrator, signal_evalues

# Window sizes are in *observations* of that signal. For 10-step layer stats
# 100 obs = 1000 optimizer steps of context.
DEFAULT_SIGNALS = {
    "loss": SignalSpec("loss", log=True, resid_window=200, conf_window=2000),
    "clip": SignalSpec("clip", log=False, resid_window=200, conf_window=2000),
    "gnorm": SignalSpec("gnorm", log=True, resid_window=100, conf_window=1000),
    "gmax": SignalSpec("gmax", log=True, resid_window=100, conf_window=1000),
    "uratio": SignalSpec("uratio", log=True, resid_window=100, conf_window=1000),
    "stale": SignalSpec("stale", log=True, resid_window=100, conf_window=1000),
    "curv": SignalSpec("curv", log=False, resid_window=100, conf_window=1000),
}


@dataclass
class Scored:
    resid: dict[str, np.ndarray]
    pvals: dict[str, np.ndarray]
    evals: dict[str, np.ndarray]
    armed: np.ndarray
    signals: list[str] = field(default_factory=list)


def score(df: pd.DataFrame, signals: list[str], cal, burn_in: int = 2000,
          specs: dict[str, SignalSpec] | None = None, seed: int = 0) -> Scored:
    """`cal` is one calibrator for all signals or a dict signal -> calibrator."""
    specs = specs or DEFAULT_SIGNALS
    armed = df["step"].to_numpy() >= burn_in
    R, P, E = {}, {}, {}
    for i, s in enumerate(signals):
        x = df[s].to_numpy(float) if s in df else np.full(len(df), np.nan)
        if np.isfinite(x).sum() < specs[s].conf_window + specs[s].resid_window:
            continue  # signal not logged for this series
        c = cal[s] if isinstance(cal, dict) else cal
        r, p, e = signal_evalues(x, specs[s], c, seed=seed + i)
        e = np.where(armed, e, 1.0)
        R[s], P[s], E[s] = r, p, e
    return Scored(R, P, E, armed, list(E))


def detect(sc: Scored, threshold: float, signals: list[str] | None = None, lambdas=DEFAULT_LAMBDAS,
           weights: dict[str, float] | None = None):
    sig = [s for s in (signals or sc.signals) if s in sc.evals]
    obs = {s: np.isfinite(sc.pvals[s]) | ~sc.armed for s in sig}
    e = combine({s: sc.evals[s] for s in sig}, weights, observed=obs)
    logR, alarms = sr_statistic(e, lambdas=lambdas, threshold=threshold)
    return logR, alarms, e


def recalibrate(sc: Scored, cal) -> Scored:
    """Re-derive e-values from cached p-values under a different calibrator."""
    E = {}
    for s, p in sc.pvals.items():
        c = cal[s] if isinstance(cal, dict) else cal
        E[s] = np.where(sc.armed, c(p), 1.0)
    return Scored(sc.resid, sc.pvals, E, sc.armed, list(E))
