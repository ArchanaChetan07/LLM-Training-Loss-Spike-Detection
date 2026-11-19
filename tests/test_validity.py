"""The guarantees are theorems under H0; these tests check the implementation
actually delivers them, and that power exists under an alternative."""

import math

import numpy as np
import pytest

from spikewatch.detector import exceedance_cs, sr_statistic, sr_statistic_loop
from spikewatch.evalues import SignalSpec, TailCalibrator, conformal_pvalues, robust_residual, signal_evalues
from spikewatch.labels import label_spikes
from spikewatch.metrics import evaluate


def test_vectorised_sr_matches_reference():
    rng = np.random.default_rng(0)
    e = rng.exponential(1.0, 3000) * (rng.random(3000) < 0.5) * 2
    e[1500:1510] = 50
    for thr in (None, 1e3, 1e6):
        a, al_a = sr_statistic(e, threshold=thr, chunk=700)
        b, al_b = sr_statistic_loop(e, threshold=thr)
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-7)
        np.testing.assert_array_equal(al_a, al_b)


def test_conformal_pvalues_uniform_under_iid():
    rng = np.random.default_rng(1)
    r = rng.standard_t(3, 60000)  # heavy tails: ranks do not care
    p = conformal_pvalues(r, 500, rng)
    p = p[np.isfinite(p)]
    for u in (0.001, 0.01, 0.1, 0.5):
        assert abs(np.mean(p <= u) - u) < 4 * math.sqrt(u * (1 - u) / len(p)) + 1e-4


def test_conformal_handles_ties():
    rng = np.random.default_rng(2)
    r = (rng.random(20000) < 0.02).astype(float)  # mostly zeros, like clip rate
    p = conformal_pvalues(r, 1000, rng)
    p = p[np.isfinite(p)]
    assert abs(np.mean(p <= 0.01) - 0.01) < 0.003


def test_calibrator_is_e_value():
    cal = TailCalibrator()
    rng = np.random.default_rng(3)
    assert abs(cal(rng.random(2_000_000)).mean() - 1.0) < 0.04  # sd ~0.009
    assert cal(np.array([np.nan]))[0] == 1.0


@pytest.mark.parametrize("mode", ["run", "rate"])
def test_false_alarm_guarantees_hold_under_iid_null(mode):
    """G-run: P(any alarm in T) <= alpha. G-rate: E[#alarms per W] <= alpha."""
    rng = np.random.default_rng(4)
    alpha, T, W, reps = 0.2, 4000, 1000, 150
    cal = TailCalibrator(betas=(0.02, 0.005))
    thr = T / alpha if mode == "run" else W / alpha
    hits, n_alarms = 0, 0
    for _ in range(reps):
        p = rng.random(T)
        _, al = sr_statistic(cal(p), threshold=thr)
        hits += al.size > 0
        n_alarms += al.size
    if mode == "run":
        assert hits / reps <= alpha + 3 * math.sqrt(alpha * (1 - alpha) / reps)
    else:
        assert n_alarms / (reps * T / W) <= alpha * 1.3


def test_power_under_tail_burst():
    rng = np.random.default_rng(5)
    p = rng.random(5000)
    p[3000:3006] = rng.random(6) * 0.001  # six extreme observations
    _, al = sr_statistic(TailCalibrator()(p), threshold=5000 / 0.05)
    assert al.size and 3000 <= al[0] <= 3006


def test_autocorrelation_breaks_rho_one():
    """Why rho exists: an AR(1) residual stream is not exchangeable, tail
    events cluster, and rho = 1 over-alarms. This is the failure mode the
    held-out audit measures on real telemetry."""
    rng = np.random.default_rng(6)
    T = 40000
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = 0.97 * x[t - 1] + rng.standard_normal()
    spec = SignalSpec("x", log=False, resid_window=100, conf_window=1000)
    _, p, e = signal_evalues(x, spec, TailCalibrator(rho=1.0))
    _, al = sr_statistic(e, threshold=1000 / 0.05)
    rate = al.size / (T / 1000)
    assert rate > 0.05  # violates G-rate's nominal 0.05 per 1000


def test_exceedance_cs_covers():
    rng = np.random.default_rng(7)
    p = rng.random(3000)
    ci = exceedance_cs(p, 0.05, 0, 2999)
    assert ci.lo <= 0.05 <= ci.hi
    p[2000:] = rng.random(1000) * 0.2  # 25% of p<=0.05 after 2000
    ci = exceedance_cs(p, 0.05, 2000, 2999)
    assert ci.lo > 0.05


def test_robust_residual_is_causal():
    x = np.r_[np.ones(300), 100.0, np.ones(10)]
    r = robust_residual(x + 1e-3 * np.arange(311) % 7, 100)
    r2 = robust_residual(np.r_[x[:300], 1.0, x[301:]] + 1e-3 * np.arange(311) % 7, 100)
    np.testing.assert_array_equal(r[:300], r2[:300])  # the future never leaks backwards


def test_labeler_finds_spike_and_recovery():
    import pandas as pd
    rng = np.random.default_rng(8)
    n = 6000
    loss = 3.0 - 0.0001 * np.arange(n) + 0.02 * rng.standard_normal(n)
    loss[4000:4030] += np.linspace(1.0, 0.0, 30)
    df = pd.DataFrame({"loss": loss, "run": "r", "step": np.arange(n)})
    eps = label_spikes(df)
    assert len(eps) == 1 and eps[0].onset == 4000 and not eps[0].diverged
    res = evaluate(np.array([3990, 100, 5500]), eps, n, horizon=200)
    assert res.early == 1 and res.lead_onset[0] == 10 and res.false_alarms == 2


def test_observed_weighting_keeps_validity():
    """Averaging only signals observed at t (predictable schedule) is still an e-value."""
    from spikewatch.detector import combine
    rng = np.random.default_rng(9)
    T = 400_000
    cal = TailCalibrator()
    a = cal(rng.random(T))
    pb = np.full(T, np.nan); pb[::10] = rng.random(T // 10)
    b = cal(pb)
    e = combine({"a": a, "b": b}, observed={"a": np.ones(T, bool), "b": np.isfinite(pb)})
    assert abs(e.mean() - 1) < 0.03
    _, al = sr_statistic(e, threshold=1000 / 0.05)
    assert al.size / (T / 1000) <= 0.05 * 1.5
