"""Ground-truth loss-spike episodes.

Labelling is offline and may look at the future; detection never sees labels.

A step t *opens* an episode when its loss exceeds the causal baseline
B_t = median(loss_{t-W..t-1}) by more than BOTH
    k_sigma * 1.4826 * MAD(loss_{t-W..t-1})    (statistically extreme), and
    rel * B_t                                  (practically large).
The episode's baseline is then frozen at B_{t*} (a long divergence must not
end just because the rolling median caught up with it). It closes after
`recover` consecutive steps with loss <= B_{t*} + max(3 sigma*, 1% B_{t*}).

Per episode we record:
    onset  first step over the threshold              -> lead time is measured to this
    peak   argmax loss in the episode                 -> "warning before peak damage"
    end    last step before recovery
    severity  (peak loss - B) / B
    diverged  never recovered inside its run (training was rewound / killed)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass
class Episode:
    onset: int      # stream position
    peak: int
    end: int
    baseline: float
    severity: float
    diverged: bool
    run: str
    step: int       # optimizer step at onset


def label_spikes(df: pd.DataFrame, window: int = 200, k_sigma: float = 8.0, rel: float = 0.05,
                 recover: int = 20, burn_in: int = 2000, min_hist: int = 100) -> list[Episode]:
    loss = df["loss"].to_numpy(float)
    s = pd.Series(loss).shift(1)
    med = s.rolling(window, min_periods=min_hist).median().to_numpy()
    mad = (s - med).abs().rolling(window, min_periods=min_hist).median().to_numpy() * 1.4826
    run = df["run"].to_numpy()
    steps = df["step"].to_numpy()
    eligible = steps >= burn_in
    eps: list[Episode] = []
    t, T = 0, len(loss)
    while t < T:
        if not (eligible[t] and np.isfinite(med[t]) and np.isfinite(loss[t])):
            t += 1
            continue
        thr = max(k_sigma * mad[t], rel * med[t])
        if loss[t] - med[t] <= thr:
            t += 1
            continue
        onset, B, sig = t, med[t], mad[t]
        band = B + max(3 * sig, 0.01 * B)
        ok_run, last_bad, u = run[t], t, t + 1
        while u < T and run[u] == ok_run and u - last_bad <= recover:
            if loss[u] > band:
                last_bad = u
            u += 1
        diverged = (u - last_bad) <= recover  # ran out of run before recovering
        seg = slice(onset, last_bad + 1)
        peak = onset + int(np.nanargmax(loss[seg]))
        eps.append(Episode(onset, peak, last_bad, float(B), float((loss[peak] - B) / B),
                           bool(diverged), str(ok_run), int(steps[onset])))
        t = last_bad + 1
    return eps


def episodes_frame(eps: list[Episode]) -> pd.DataFrame:
    return pd.DataFrame([asdict(e) for e in eps])


def label_tiers(df: pd.DataFrame, burn_in: int = 10000, **kw):
    """(major, minor) episodes. Major: >= 8 sigma and >= 5% above baseline.
    Minor: >= 6 sigma and >= 2%, not overlapping a major one. Minor episodes
    are 'don't care' in evaluation: real but sub-threshold loss bumps that a
    detector may legitimately flag."""
    major = label_spikes(df, burn_in=burn_in, **kw)
    cand = label_spikes(df, k_sigma=6.0, rel=0.02, burn_in=burn_in, **kw)
    occ = np.zeros(len(df) + 1, bool)
    for e in major:
        occ[e.onset: e.end + 1] = True
    minor = [e for e in cand if not occ[e.onset: e.end + 1].any()]
    return major, minor
