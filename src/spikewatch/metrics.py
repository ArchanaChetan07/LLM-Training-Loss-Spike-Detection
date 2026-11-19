"""Lead time and false-alarm accounting.

For an episode with onset t*, an alarm at position a is
    early     if  t* - H <= a <  t*         (lead = t* - a  >= 1)
    at onset  if  a == t*                   (lead = 0)
    late      if  t* <  a <= end            (lead < 0; still caught before recovery)
and the episode's alarm is the *earliest* one in [t* - H, end].

An alarm outside every [t* - H, end + grace] is a false alarm. False-alarm
rates are normalised by *stable* steps (armed, outside those intervals), and
also reported as the fraction of disjoint W-step stable windows that contain
at least one false alarm (the quantity G-rate bounds by alpha_W).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EvalResult:
    n_events: int
    detected: int          # any alarm in [t*-H, end]
    early: int             # lead >= 1
    lead_onset: np.ndarray  # per event, NaN if missed
    lead_peak: np.ndarray
    false_alarms: int
    stable_steps: int
    fa_per_10k: float
    windows: int
    windows_with_fa: int

    @property
    def window_fa_rate(self) -> float:
        return self.windows_with_fa / self.windows if self.windows else float("nan")

    def summary(self) -> dict:
        lo = self.lead_onset[np.isfinite(self.lead_onset)]
        lp = self.lead_peak[np.isfinite(self.lead_peak)]
        return {
            "events": self.n_events,
            "detected": self.detected,
            "detected_ahead_of_onset": self.early,
            "detected_before_peak": int(np.sum(lp >= 1)),
            "median_lead_onset": float(np.median(lo)) if lo.size else float("nan"),
            "median_lead_peak": float(np.median(lp)) if lp.size else float("nan"),
            "false_alarms": self.false_alarms,
            "fa_per_10k": self.fa_per_10k,
            "window_fa_rate": self.window_fa_rate,
            "stable_steps": self.stable_steps,
        }


def evaluate(alarms: np.ndarray, episodes, n: int, armed: np.ndarray | None = None, horizon: int = 200,
             grace: int = 200, window: int = 1000, ignore=()) -> EvalResult:
    """`ignore`: minor (sub-threshold) episodes. Alarms within
    [onset - horizon, end + grace] of one are neither hits nor false alarms,
    and those steps do not count as stable."""
    alarms = np.sort(np.asarray(alarms, dtype=int))
    armed = np.ones(n, bool) if armed is None else np.asarray(armed, bool)
    alarms = alarms[armed[alarms]] if alarms.size else alarms
    excl = np.zeros(n, bool)
    lead_o, lead_p = [], []
    prev_end = -1
    for ep in sorted(episodes, key=lambda e: e.onset):
        # an alarm raised for the previous episode (or during its aftermath)
        # must not be credited as an early warning for this one
        lo = max(0, ep.onset - horizon, prev_end + grace + 1)
        lo = min(lo, ep.onset)
        prev_end = ep.end
        excl[lo: min(n, ep.end + grace + 1)] = True
        a = alarms[(alarms >= lo) & (alarms <= ep.end)]
        if a.size:
            lead_o.append(ep.onset - a[0]); lead_p.append(ep.peak - a[0])
        else:
            lead_o.append(np.nan); lead_p.append(np.nan)
    lead_o, lead_p = np.asarray(lead_o, float), np.asarray(lead_p, float)
    for ep in ignore:
        excl[max(0, ep.onset - horizon): min(n, ep.end + grace + 1)] = True
    stable = armed & ~excl
    fa = alarms[stable[alarms]] if alarms.size else alarms
    # disjoint fully-stable windows
    is_fa = np.zeros(n, bool); is_fa[fa] = True
    windows = wfa = 0
    run_start = None
    for i in range(n + 1):
        if i < n and stable[i]:
            if run_start is None:
                run_start = i
            continue
        if run_start is not None:
            L = i - run_start
            k = L // window
            if k:
                blk = is_fa[run_start: run_start + k * window].reshape(k, window).any(1)
                windows += k; wfa += int(blk.sum())
            run_start = None
    ss = int(stable.sum())
    return EvalResult(len(episodes), int(np.isfinite(lead_o).sum()), int(np.sum(lead_o >= 1)), lead_o, lead_p,
                      int(fa.size), ss, 1e4 * fa.size / max(ss, 1), windows, wfa)
