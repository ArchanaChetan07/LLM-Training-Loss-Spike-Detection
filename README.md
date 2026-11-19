# spikewatch — loss-spike early warning with formal false-alarm guarantees

Large-model pretraining runs lose days to loss spikes. Today's monitors are thresholds
("alarm if grad-norm > 7σ") tuned by eye on one run. They come with no statement about how
often they will page someone for nothing, and that calibration silently breaks on the next run.
spikewatch treats spike detection as **sequential hypothesis testing on training telemetry**,
using anytime-valid inference (e-values, e-processes, confidence sequences). The false-alarm
rate is then a theorem under a stated null, not a tuning outcome. The whole evaluation is
pre-registered and run on AI2's public OLMo training logs.

## What is guaranteed, under which null

At each logged step `t`, every signal `s` (loss, clip rate, grad norm, max per-tensor grad
ratio, Adam update ratio, second-moment staleness, secant curvature) is turned into:

1. A causal robust residual: median/MAD over the trailing window.
2. A **predictable conformal p-value**: the residual's randomized upper-tail rank among its own
   trailing history.
3. A **tail e-value**, from the bounded calibrator `f(p) = Σ_k w_k 1[p ≤ β_k] / (ρ_k(state) β_k)`
   with β = (1%, 0.3%, 0.1%).

The e-values are averaged over the signals observed at `t`. A convex average of e-values is an
e-value, so no independence between signals is assumed. They then feed a **mixture
Shiryaev–Roberts e-detector**:

```
R_t = mean_λ R_t^λ,   R_t^λ = (R_{t-1}^λ + 1)(1 − λ + λ e_t),   λ ∈ {0.05, 0.2, 0.5, 0.8}
```

Under H0, `E[R_t | F_{t-1}] ≤ R_{t-1} + 1`. Two guarantees follow (proofs in
[`detector.py`](src/spikewatch/detector.py); both are checked by simulation in
[`tests/test_validity.py`](tests/test_validity.py)):

| | threshold | guarantee under H0 |
|---|---|---|
| **G-rate** (primary) | `W/α_W`, restart after alarm | E[# false alarms per W steps] ≤ α_W ⇒ P(a W-step window has a false alarm) ≤ α_W |
| **G-run** | `T/α` | P(any false alarm in a T-step run) ≤ α (Ville's inequality on `R_t/T + 1 − t/T`) |

Every alarm also has a **confidence-sequence reading**. With the tail calibrator, each
change-point component of `R` is the betting capital of the Waudby-Smith–Ramdas confidence
sequence for the exceedance rate `P(p_t ≤ β)`. So an alarm states: "since step j, signal s has
been in its top-β tail at a rate whose time-uniform CS excludes ρβ". `exceedance_cs` reports
that interval.

**The null that matters.** Exact exchangeability (ρ = 1) is false for training telemetry,
because tail events cluster. Pre-registration pilot: ρ = 1 gave a 7.1% window false-alarm rate
at nominal 5%. spikewatch therefore tests a **cluster-aware null** instead: conditional tail
inflation is allowed up to `ρ_k(s)`, where the state `s` is "any p ≤ 1% among the last 5
observations of this signal". `ρ_k(s)` is fitted once, on OLMo-1B steps 10k–250k, using 95%
Clopper–Pearson upper bounds. It is then frozen and applied to every other model unchanged.
The guarantee is exact given ρ; **whether ρ transfers across models is an empirical question,
and it is the central test of this project.**

## Protocol (pre-registered)

[`docs/PREREGISTRATION.md`](docs/PREREGISTRATION.md) was frozen before any held-out series was
scored. Its two deviations are logged in that file, with reasons.

* **Labels.**
  * Major spike: loss > causal median(200) + max(8σ_MAD, 5%). The episode baseline is frozen
    at onset; the episode ends after 20 recovered steps.
  * Minor spike (6σ, 2%): treated as don't-care.
* **Scoring.**
  * Detection: an alarm in [onset − 200, end]. Lead is measured to onset and also to peak.
  * Detection windows never reach back into the previous episode.
  * False alarm: an alarm outside every episode window.
  * Window FA rate: the fraction of disjoint, fully stable 1000-step windows that contain ≥ 1
    false alarm.
* **Splits.**
  * Calibration: OLMo-1B steps 10k–250k.
  * Held out:
    * OLMo-1B steps > 250k
    * OLMo-7B
    * OLMo-1.7-7B
    * OLMo-2-1124-7B stage 1
  * Own runs: ρ is fitted on 5 control runs and tested on 24 unstable ones.
* **Baselines.** Max robust-z threshold (including the OLMo-2 "7σ" rule) and Page CUSUM. Both
  are tuned on the calibration split to the most sensitive setting with FA ≤ 5%, then frozen.
* **Endpoint.**
  * Success requires both:
    * window FA ≤ 5% on **every** held-out OLMo series;
    * ≥ 50% of pooled held-out spikes flagged ≥ 1 step before onset.
  * Falsified if FA > 5% on any held-out OLMo series. That outcome is reported as such, not
    re-tuned.

## Status (2026-11-19)

The evaluation is **in progress**. Numbers are generated into [`results/RESULTS.md`](results/RESULTS.md) by `scripts/make_report.py`.

* Done:
  * calibration (OLMo-1B steps 10k–250k);
  * held-out OLMo-1B steps > 250k: no major spikes; e-detector window false-alarm rate 3.8%
    vs 10.6% for the z-threshold tuned on the calibration split;
  * own-runs arm: 1 major spike in 24 unstable runs, detected 3 steps before onset; window
    false-alarm rate 9.8% (4/41 windows, binomial p = 0.15 vs 5%).
* Pending: final scoring of OLMo-7B, OLMo-1.7-7B and OLMo-2-7B (telemetry download finishing).
  The figures currently in `figures/` include partial 7B data and will be regenerated.


## Reproduce

```powershell
python -m pip install -e .[fetch,train,dev]
python -m pytest -q                                   # validity tests (G-run, G-rate, calibrator, CS)
python scripts/fetch_olmo.py ai2-llm/OLMo-1B          # + OLMo-7B (--prefix OLMo-7B-run / OLMo-1.7-7B-run), OLMo-2-1124-7B
python -W ignore scripts/run_olmo_eval.py --calibrate # fit ρ on OLMo-1B 10k–250k, score all held-out series
bash scripts/sweep_small.sh                           # own runs (29 × 3000 steps, ~1.5 GPU-h on a T1000)
python -W ignore scripts/run_small_eval.py
python -W ignore scripts/make_figures.py
python scripts/make_report.py                         # -> results/RESULTS.md (all numbers below)
```

## Layout

| path | content |
|---|---|
| `src/spikewatch/evalues.py` | causal residuals, conformal p-values, tail / cluster-aware calibrators |
| `src/spikewatch/detector.py` | mixture SR e-detector (vectorised, log-space), thresholds, combination, exceedance CS; proofs in docstring |
| `src/spikewatch/olmo.py` | OLMo W&B telemetry → per-step signals (update ratio, staleness, curvature from per-tensor norms) |
| `src/spikewatch/labels.py`, `metrics.py` | spike/divergence labeler; lead-time / false-alarm scoring |
| `src/spikewatch/audit.py` | tail-inflation audit, ρ(s) fitting |
| `src/spikewatch/experiment.py` | the pre-registered protocol as code |
| `scripts/train_small.py` | byte-level GPT with exact per-step telemetry (update ratio, secant curvature, logit max) |

Author: Archana Suresh Patil.
