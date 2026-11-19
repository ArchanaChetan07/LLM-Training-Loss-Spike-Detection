"""spikewatch: anytime-valid early warning for loss spikes in LLM pre-training."""

from .detector import combine, exceedance_cs, sr_statistic, threshold_for
from .evalues import SignalSpec, TailCalibrator, conformal_pvalues, robust_residual, signal_evalues
from .labels import Episode, label_spikes
from .metrics import evaluate

__all__ = [
    "combine", "exceedance_cs", "sr_statistic", "threshold_for", "SignalSpec", "TailCalibrator",
    "conformal_pvalues", "robust_residual", "signal_evalues", "Episode", "label_spikes", "evaluate",
]
__version__ = "0.1.0"
