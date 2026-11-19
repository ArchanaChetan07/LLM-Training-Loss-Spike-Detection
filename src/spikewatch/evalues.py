"""Predictable conformal p-values and p-to-e calibrators.

Pipeline per telemetry signal x_t (all steps strictly causal):

  1. residual   r_t = (x_t - med(x_{t-w..t-1})) / (1.4826 MAD(x_{t-w..t-1}))
  2. p-value    p_t = rank of r_t among the previous n residuals (randomized
                conformal, upper tail): p_t = (#{r_i > r_t} + U (#{r_i = r_t} + 1)) / (n + 1)
  3. e-value    e_t = f(p_t) with f a calibrator (decreasing, int_0^1 f = 1)

The null hypothesis the whole system is built on:

  H0(rho):  P(p_t <= u | F_{t-1}) <= rho * u   for every u in the calibrator's
            support and every t (conditional super-uniformity up to a factor rho).

rho = 1 is the textbook assumption; it holds exactly when the residual stream
is exchangeable within the conformal window. Training telemetry is
autocorrelated, so we carry an explicit robustness margin rho >= 1 and
*measure* it on held-out stable stretches (see audit.py) instead of assuming it.
Under H0(rho), e_t = f(p_t) / rho is an e-value: E[e_t | F_{t-1}] <= 1.

The calibrator we use is a finite mixture of tail indicators,

    f(p) = sum_k w_k 1[p <= beta_k] / beta_k ,   sum_k w_k = 1,

whose expectation under a super-uniform p is <= 1. It is bounded (so betting
on it is safe), and it only uses *whether* a residual is in the extreme tail
of its own recent history, never its magnitude: a heavy-tailed gradient-norm
outlier cannot buy more evidence than its rank. That is what makes the
guarantee robust to the fat tails of training telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MAD_TO_SIGMA = 1.4826


def robust_residual(x: np.ndarray, window: int, min_periods: int | None = None) -> np.ndarray:
    """Causal robust z-score: centre and scale use x_{t-window..t-1} only."""
    s = pd.Series(np.asarray(x, dtype=float))
    mp = min_periods or max(10, window // 4)
    past = s.shift(1)
    med = past.rolling(window, min_periods=mp).median()
    mad = (past - med).abs().rolling(window, min_periods=mp).median()
    # a MAD of exactly 0 (flat stretches, e.g. clip-rate == 0) would make every
    # positive move infinitely surprising; floor it at a small fraction of |median|
    floor = 1e-3 * med.abs() + 1e-12
    scale = MAD_TO_SIGMA * np.maximum(mad.to_numpy(), floor.to_numpy())
    return (s.to_numpy() - med.to_numpy()) / scale


def conformal_pvalues(r: np.ndarray, n: int, rng: np.random.Generator, min_hist: int | None = None,
                      chunk: int = 4096) -> np.ndarray:
    """Upper-tail randomized conformal p-values against the previous n finite
    residuals (NaNs are skipped and yield NaN p-values).

    Vectorised over a sliding window; O(T n) work, chunked for memory.
    """
    r = np.asarray(r, dtype=float)
    out = np.full(r.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(r))
    v = r[ok]
    m = len(v)
    min_hist = n if min_hist is None else min_hist
    if m <= min_hist:
        return out
    u = rng.random(m)
    p = np.full(m, np.nan)
    # windows of the previous n points: for index i (>= min_hist) the history
    # is v[max(0, i-n): i]
    for lo in range(min_hist, m, chunk):
        hi = min(m, lo + chunk)
        idx = np.arange(lo, hi)
        start = np.maximum(0, idx - n)
        if np.all(idx - start == n):
            win = np.lib.stride_tricks.sliding_window_view(v[lo - n: hi - 1], n)  # (hi-lo, n)
            cur = v[lo:hi, None]
            gt = (win > cur).sum(1)
            eq = (win == cur).sum(1)
            k = np.full(hi - lo, n)
        else:  # ragged warm-up region (only when min_hist < n)
            gt = np.empty(hi - lo); eq = np.empty(hi - lo); k = idx - start
            for j, i in enumerate(idx):
                w = v[start[j]: i]
                gt[j] = (w > v[i]).sum(); eq[j] = (w == v[i]).sum()
        p[lo - 0: hi] = (gt + u[lo:hi] * (eq + 1)) / (k + 1)
    out[ok] = p
    return out


@dataclass(frozen=True)
class TailCalibrator:
    """f(p) = sum_k w_k 1[p <= beta_k] / beta_k, divided by the margin rho."""

    betas: tuple[float, ...] = (0.01, 0.003, 0.001)
    weights: tuple[float, ...] | None = None
    rho: float = 1.0

    def __post_init__(self):
        if self.rho < 1:
            raise ValueError("rho must be >= 1")
        w = self.weights or tuple([1.0 / len(self.betas)] * len(self.betas))
        if abs(sum(w) - 1) > 1e-9 or len(w) != len(self.betas):
            raise ValueError("weights must sum to 1 and match betas")
        object.__setattr__(self, "weights", w)

    def __call__(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        e = np.zeros_like(p)
        for b, w in zip(self.betas, self.weights):
            e += w * (p <= b) / b
        e /= self.rho
        e[~np.isfinite(p)] = 1.0  # missing observation: a neutral e-value
        return e

    @property
    def max_e(self) -> float:
        return float(sum(w / b for b, w in zip(self.betas, self.weights)) / self.rho)


def recent_tail_state(p: np.ndarray, beta: float, K: int) -> np.ndarray:
    """Predictable state s_t = 1 if any of the previous K *observed* p-values
    were <= beta. Unobserved steps inherit the state of the next observation."""
    p = np.asarray(p, float)
    obs = np.flatnonzero(np.isfinite(p))
    tail = p[obs] <= beta
    rec = np.zeros(tail.shape, bool)
    for k in range(1, K + 1):
        rec[k:] |= tail[:-k]
    s = np.zeros(p.shape, np.int8)
    s[obs] = rec
    return s


@dataclass(frozen=True)
class StatefulCalibrator:
    """Cluster-aware calibrator for the null

        H0:  P(p_t <= beta_k | F_{t-1}) <= rho_k(s_t) beta_k,   s_t in {0, 1},

    with s_t = recent_tail_state(p, max(betas), K). Each e-value is
    e_t = sum_k w_k 1[p_t <= beta_k] / (beta_k rho_k(s_t)), valid under H0.

    Real telemetry has tail clusters (a hard data shard, an LR-schedule
    kink); rho_k(1) >> 1 says 'a second tail event right after a first one is
    normal', so only bursts *larger than normal* accumulate evidence.
    rho is estimated once, on calibration data, by audit.fit_stateful.
    """

    betas: tuple[float, ...]
    rho0: tuple[float, ...]
    rho1: tuple[float, ...]
    K: int = 5
    weights: tuple[float, ...] | None = None

    def __call__(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, float)
        w = self.weights or [1.0 / len(self.betas)] * len(self.betas)
        s = recent_tail_state(p, max(self.betas), self.K)
        e = np.zeros_like(p)
        for b, wk, r0, r1 in zip(self.betas, w, self.rho0, self.rho1):
            rho = np.where(s == 1, r1, r0)
            e += wk * (p <= b) / (b * rho)
        e[~np.isfinite(p)] = 1.0
        return e


@dataclass
class SignalSpec:
    name: str
    log: bool = True          # residuals in log space for positive scale quantities
    resid_window: int = 100   # observations used for the robust centre/scale
    conf_window: int = 1000   # observations used for the conformal rank
    two_sided: bool = False   # loss *drops* are not alarming; grad-norm collapse can be
    meta: dict = field(default_factory=dict)


def signal_evalues(x: np.ndarray, spec: SignalSpec, cal: TailCalibrator, seed: int = 0):
    """x (with NaN where unobserved) -> (residual, p-value, e-value) arrays."""
    x = np.asarray(x, dtype=float)
    obs = np.isfinite(x)
    xv = x[obs]
    if spec.log:
        xv = np.log(np.maximum(xv, 1e-30))
    r_obs = robust_residual(xv, spec.resid_window)
    if spec.two_sided:
        r_obs = np.abs(r_obs)
    rng = np.random.default_rng(seed)
    p_obs = conformal_pvalues(r_obs, spec.conf_window, rng)
    r = np.full(x.shape, np.nan); r[obs] = r_obs
    p = np.full(x.shape, np.nan); p[obs] = p_obs
    return r, p, cal(p)
