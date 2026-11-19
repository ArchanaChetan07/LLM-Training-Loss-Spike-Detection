"""Anytime-valid change detection on e-value streams (e-detectors).

Given per-step e-values e_t (E[e_t | F_{t-1}] <= 1 under H0), we bet a fraction
lambda of our wealth on each one. For a candidate change point j the wealth is

    M_t^(j) = prod_{i=j..t} (1 - lambda + lambda e_i),

a nonnegative supermartingale under H0 that starts at 1 at time j. We do not
know j, so we sum over all candidate starts (Shiryaev-Roberts style):

    R_t = sum_{j<=t} M_t^(j)  =  (R_{t-1} + 1) (1 - lambda + lambda e_t),

an O(1) recursion. Mixing over a grid of lambdas (average of R^lambda) keeps
every property below and removes the need to tune lambda.

Two guarantees, both from the single fact that R_t - t is a supermartingale
under H0 (E[R_t | F_{t-1}] <= R_{t-1} + 1):

  (G-run)  Raise an alarm when R_t >= T / alpha, T the planned run length.
           Then P_H0(any false alarm in the first T steps) <= alpha.
           Proof: S_t = R_t / T + (1 - t/T) is a nonnegative supermartingale
           with S_0 = 1 (it is the uniform-prior mixture over change points);
           Ville's inequality gives P(sup_t S_t >= 1/alpha) <= alpha.

  (G-rate) Raise an alarm when R_t >= c and restart (R <- 0). Then
           E_H0[# false alarms in T steps] <= T / c.
           Proof: R_t + c N_t - t is a supermartingale (a restart discards
           R_tau >= c, which c N_t pays for). With c = W / alpha_W, the
           expected number of false alarms in any W-step window is <= alpha_W,
           hence P(window contains a false alarm) <= alpha_W by Markov.

Duality with confidence sequences: with the tail calibrator 1[p<=beta]/beta,
each M^(j) is exactly the betting capital process of the Waudby-Smith--Ramdas
confidence sequence for the exceedance rate q = P(p_t <= beta) since j, tested
at q = beta. An alarm is the statement "for some recent start j, the
time-uniform CS for the tail-exceedance rate has left [0, rho*beta]".
`exceedance_cs` reports that interval at alarm time for interpretability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

DEFAULT_LAMBDAS = (0.05, 0.2, 0.5, 0.8)


def sr_statistic_loop(e: np.ndarray, lambdas=DEFAULT_LAMBDAS, threshold: float | None = None,
                      restart: bool = True):
    """Reference implementation (O(T) Python loop). Used by the tests to pin
    the vectorised version below."""
    e = np.asarray(e, dtype=float)
    lam = np.asarray(lambdas, dtype=float)
    logR = np.full(len(lam), -np.inf)
    out = np.empty(len(e))
    alarms = []
    logc = math.log(threshold) if threshold else np.inf
    for t, et in enumerate(e):
        logR = np.logaddexp(logR, 0.0) + np.log1p(lam * (et - 1.0))
        m = np.max(logR)
        lr = m + math.log(np.mean(np.exp(logR - m)))
        out[t] = lr
        if lr >= logc:
            alarms.append(t)
            if restart:
                logR[:] = -np.inf
    return out, np.asarray(alarms, dtype=int)


def _logmeanexp(a, axis=0):
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return np.squeeze(m, axis) + np.log(np.mean(np.exp(a - m), axis=axis))


def sr_statistic(e: np.ndarray, lambdas=DEFAULT_LAMBDAS, threshold: float | None = None,
                 restart: bool = True, chunk: int = 50000):
    """Mixture Shiryaev-Roberts e-detector, vectorised.

    Closed form of R_t = (R_{t-1} + 1) L_t from a state R_0 at offset s:
        R_t = P_t (R_0 + sum_{j=s..t} 1 / P_{j-1}),  P_t = prod_{i=s..t} L_i,
    evaluated in log space with logaddexp.accumulate. On an alarm (with
    restart) the state is reset and the closed form re-anchored there.

    Returns (log R_t array, alarm indices).
    """
    e = np.asarray(e, dtype=float)
    lam = np.asarray(lambdas, dtype=float)[:, None]
    T = len(e)
    out = np.empty(T)
    alarms = []
    logc = math.log(threshold) if threshold else np.inf
    logR0 = np.full((lam.shape[0], 1), -np.inf)
    s = 0
    while s < T:
        stop = min(T, s + chunk)
        inc = np.log1p(lam * (e[None, s:stop] - 1.0))          # (L, n)
        logP = np.cumsum(inc, axis=1)
        logP_prev = np.concatenate([np.zeros((lam.shape[0], 1)), logP[:, :-1]], axis=1)
        acc = np.logaddexp.accumulate(-logP_prev, axis=1)
        logR = logP + np.logaddexp(logR0, acc)
        lr = _logmeanexp(logR, axis=0)
        hit = np.flatnonzero(lr >= logc)
        if hit.size and restart:
            k = hit[0]
            out[s: s + k + 1] = lr[: k + 1]
            alarms.append(s + k)
            logR0 = np.full_like(logR0, -np.inf)
            s = s + k + 1
            continue
        out[s:stop] = lr
        alarms.extend((s + hit).tolist())
        logR0 = logR[:, -1:]
        s = stop
    return out, np.asarray(alarms, dtype=int)


def threshold_for(mode: str, alpha: float, horizon: int) -> float:
    """mode='run': P(any false alarm within `horizon` steps) <= alpha.
    mode='rate': E[# false alarms per `horizon` steps] <= alpha."""
    if mode not in ("run", "rate"):
        raise ValueError(mode)
    return horizon / alpha


def combine(evals: dict[str, np.ndarray], weights: dict[str, float] | None = None,
            observed: dict[str, np.ndarray] | None = None) -> np.ndarray:
    """Weighted average of e-values across signals: valid without any
    independence assumption (a convex combination of e-values is an e-value).
    Products would be more powerful but need conditional independence across
    signals, which gradient norm, update ratio and loss obviously violate.

    If `observed` masks are given, step t averages only the signals observed
    at t. Valid because the logging schedule (loss every step, layer stats
    every 10) is fixed in advance, so the weights are predictable; it stops
    unlogged signals from diluting the evidence with neutral e = 1 values."""
    names = list(evals)
    w = weights or {k: 1.0 / len(names) for k in names}
    if observed is None:
        tot = sum(w[k] for k in names)
        return sum(w[k] / tot * np.asarray(evals[k], dtype=float) for k in names)
    num = sum(w[k] * np.where(observed[k], evals[k], 0.0) for k in names)
    den = sum(w[k] * observed[k].astype(float) for k in names)
    return np.where(den > 0, num / np.maximum(den, 1e-300), 1.0)


@dataclass
class CSInterval:
    start: int
    n: int
    exceed: int
    lo: float
    hi: float


def exceedance_cs(p: np.ndarray, beta: float, start: int, end: int, alpha: float = 0.05,
                  grid: int = 400) -> CSInterval:
    """Betting confidence sequence (hedged capital, Waudby-Smith & Ramdas 2023)
    for q = P(p_t <= beta) using observations start..end, evaluated at `end`.
    Valid simultaneously over all end times. Used to *explain* an alarm:
    'since step j, signal s has been in its top beta tail at rate in [lo, hi]'."""
    x = (np.asarray(p[start: end + 1]) <= beta)[np.isfinite(p[start: end + 1])].astype(float)
    qs = np.linspace(1e-4, 1 - 1e-4, grid)
    logK_up = np.zeros(grid); logK_dn = np.zeros(grid)
    s1 = 0.5; s2 = 0.25; n = 0
    for xi in x:  # predictable plug-in bets (aGRAPA-lite)
        mu = s1 / (n + 1); var = max(s2 / (n + 1) - mu * mu, 1e-4)
        lam = np.minimum(np.sqrt(2 * math.log(2 / alpha) / (var * (n + 1) * math.log(n + 2))), 0.5)
        logK_up += np.log1p(np.minimum(lam, 0.5 / qs) * (xi - qs))
        logK_dn += np.log1p(-np.minimum(lam, 0.5 / (1 - qs)) * (xi - qs))
        s1 += xi; s2 += xi * xi; n += 1
    keep = np.logaddexp(logK_up, logK_dn) - math.log(2) < math.log(1 / alpha)
    lo, hi = (qs[keep].min(), qs[keep].max()) if keep.any() else (np.nan, np.nan)
    return CSInterval(start, n, int(x.sum()), float(lo), float(hi))
