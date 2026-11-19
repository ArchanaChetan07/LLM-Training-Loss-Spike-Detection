"""Label-free measurement of how far real telemetry is from H0(rho = 1).

The guarantee needs P(p_t <= u | F_{t-1}) <= rho u. Marginal calibration of
conformal p-values is automatic; what breaks it is *clustering*: given a tail
event in the recent past, the next tail event is more likely than u. We
estimate the worst conditional tail inflation directly,

    rho_hat(beta) = P(p_t <= beta | some p_{t-k} <= beta, k = 1..K) / beta,

on stable stretches only, and take an upper confidence bound so the estimate
is conservative. This is the only quantity fitted on calibration data; the
alarm threshold itself comes from alpha and the horizon, never from labels.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def tail_inflation(p: np.ndarray, mask: np.ndarray, beta: float = 0.01, K: int = 5, conf: float = 0.95):
    """Return dict(marginal, conditional, conditional_ucb, n_cond) for one signal.
    Only observed (finite) p-values count; lags are in observations."""
    ok = np.isfinite(p) & mask
    q = p[ok]
    if q.size < 100:
        return {"marginal": np.nan, "conditional": np.nan, "conditional_ucb": np.nan, "n_cond": 0}
    tail = q <= beta
    recent = np.zeros_like(tail)
    for k in range(1, K + 1):
        recent[k:] |= tail[:-k]
    n = int(recent.sum()); x = int((tail & recent).sum())
    ucb = stats.beta.ppf(conf, x + 1, max(n - x, 0) + 1) if n else np.nan  # Clopper-Pearson upper
    return {"marginal": float(tail.mean() / beta), "conditional": float(x / max(n, 1) / beta),
            "conditional_ucb": float(ucb / beta), "n_cond": n}


def estimate_rho(pvals: dict[str, np.ndarray], mask: np.ndarray, betas=(0.01, 0.003, 0.001), K: int = 5):
    """Per-signal table and an overall rho: the tail-weighted mean of the
    per-signal conditional UCBs (the combined e-value is an average, so its
    inflation is the average inflation), floored at 1."""
    table = {}
    for s, p in pvals.items():
        table[s] = {b: tail_inflation(p, mask, b, K) for b in betas}
    vals = [table[s][b]["conditional_ucb"] for s in table for b in betas
            if np.isfinite(table[s][b]["conditional_ucb"])]
    rho = max(1.0, float(np.mean(vals))) if vals else 1.0
    return rho, table


def fit_stateful(pvals: dict[str, np.ndarray], mask: np.ndarray, betas=(0.01, 0.003, 0.001), K: int = 5,
                 conf: float = 0.95):
    """Fit a StatefulCalibrator per signal on calibration data.

    For each signal, level beta_k and state s in {0,1} we take the
    Clopper-Pearson upper bound on P(p_t <= beta_k | s_t = s) over stable
    observations and set rho_k(s) = max(1, UCB / beta_k). Using the UCB (not
    the point estimate) and flooring at 1 keeps the fitted null conservative.
    """
    from .evalues import StatefulCalibrator, recent_tail_state

    cals, table = {}, {}
    for sname, p in pvals.items():
        st = recent_tail_state(p, max(betas), K)
        ok = np.isfinite(p) & mask
        r0, r1, rows = [], [], []
        for b in betas:
            for s, acc in ((0, r0), (1, r1)):
                sel = ok & (st == s)
                n = int(sel.sum()); x = int((p[sel] <= b).sum())
                ucb = stats.beta.ppf(conf, x + 1, n - x + 1) if n else 1.0
                acc.append(max(1.0, ucb / b))
                rows.append({"beta": b, "state": s, "n": n, "x": x, "rate_over_beta": x / max(n, 1) / b,
                             "rho_ucb": acc[-1]})
        cals[sname] = StatefulCalibrator(tuple(betas), tuple(r0), tuple(r1), K)
        table[sname] = rows
    return cals, table
