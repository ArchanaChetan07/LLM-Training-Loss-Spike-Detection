"""Classical detectors operating on the same causal robust z-scores.

None of these has a false-alarm guarantee that survives a change of run,
model size or optimizer settings: their thresholds are tuned. That is the
comparison the evaluation makes explicit (tune on one model, test on another).
"""

from __future__ import annotations

import numpy as np


def _alarms_with_refractory(trigger: np.ndarray, refractory: int) -> np.ndarray:
    idx = np.flatnonzero(trigger)
    if refractory <= 0 or idx.size == 0:
        return idx
    keep = [idx[0]]
    for i in idx[1:]:
        if i - keep[-1] > refractory:
            keep.append(i)
    return np.asarray(keep)


def zscore_alarms(z: dict[str, np.ndarray], k: float, refractory: int = 0) -> np.ndarray:
    """Alarm when max over signals of the robust z-score exceeds k
    (the 'grad norm > k sigma' rule most training stacks ship)."""
    Z = np.nanmax(np.vstack([np.nan_to_num(v, nan=-np.inf) for v in z.values()]), axis=0)
    return _alarms_with_refractory(Z >= k, refractory)


def cusum_alarms(z: dict[str, np.ndarray], h: float, drift: float = 1.0) -> np.ndarray:
    """One-sided Page CUSUM per signal on clipped z-scores, alarm on any
    signal; all statistics reset after an alarm."""
    names = list(z)
    Z = np.vstack([np.clip(np.nan_to_num(z[n], nan=0.0), -10, 10) for n in names])
    S = np.zeros(len(names))
    alarms = []
    for t in range(Z.shape[1]):
        S = np.maximum(0.0, S + Z[:, t] - drift)
        if np.any(S >= h):
            alarms.append(t)
            S[:] = 0.0
    return np.asarray(alarms, dtype=int)
