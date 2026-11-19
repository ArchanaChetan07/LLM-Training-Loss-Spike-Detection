# Pre-registration — spikewatch evaluation protocol

**Frozen 2026-11-18, before any held-out series was scored.** At freeze time the
only telemetry inspected was OLMo-1B steps 0–199k (runs 001–014, partly
downloaded), used to design the labeler, discover the two failure modes below,
and pilot the pipeline. Everything in this file is fixed from here on; any
later change is logged in the "Deviations" section with its reason.

## What was learned on the pilot (and therefore is *not* evidence)

1. Exact exchangeability (rho = 1) over-alarms on real telemetry: window
   false-alarm rate 7.1% at nominal 5% on OLMo-1B 0–190k. Cause: tail events
   cluster. -> cluster-aware null (`StatefulCalibrator`), fitted on calibration data.
2. Alarms in steps 2.6k–5k come from early-training non-stationarity -> arm at 10k.
3. A +3.9% loss bump at step 79.4k is a real sub-threshold spike -> "minor" tier = don't care.

## Data splits

| Role | Data |
|---|---|
| Calibration (fits rho(s) only) | OLMo-1B, optimizer steps 10,000 – 250,000, stable steps only |
| Held-out test 1 (same model, later training) | OLMo-1B, steps > 250,000 |
| Held-out test 2 (transfer: 7B) | OLMo-7B (v1) all runs |
| Held-out test 3 (transfer: 7B, new data mix) | OLMo-1.7-7B all runs |
| Held-out test 4 (transfer: different architecture/recipe) | OLMo-2-1124-7B stage 1 |
| Own runs | small GPT sweep; rho(s) fitted on the lr=1e-3 control runs only, tested on all others |

The OLMo calibration is applied unchanged to every OLMo test series (matched by signal name).

## Fixed method

* Signals: loss, clip, gnorm, gmax, uratio, stale, curv (definitions in `olmo.py`; missing signals skipped).
* Residuals: causal robust z (median/MAD over 200 obs for per-step signals, 100 obs for 10-step signals).
* Conformal window: 2000 obs (per-step), 1000 obs (10-step). Upper tail, randomized ties.
* Calibrator: betas (0.01, 0.003, 0.001), equal weights, state = any p <= 0.01 in last K = 5 obs,
  rho_k(s) = max(1, 95% Clopper–Pearson UCB / beta_k) on calibration stable steps.
* Signals combined by equal-weight average of e-values.
* Detector: mixture Shiryaev–Roberts, lambda in {0.05, 0.2, 0.5, 0.8}, restart after alarm.
* Armed from optimizer step 10,000 (OLMo) / 1,000 (small runs).

## Guarantees evaluated

* **Primary (G-rate)**: threshold c = W / alpha_W with W = 1000 steps, alpha_W = 0.05:
  under H0, E[#false alarms per 1000 steps] <= 0.05, hence P(a given 1000-step window has a false alarm) <= 5%.
* Secondary (G-run): threshold T / alpha with alpha = 0.05, T = series length: P(any false alarm in the run) <= 5%.

## Labels and scoring

* Major spike: loss > causal median(200) + max(8 sigma_MAD, 5%). Episode baseline frozen at onset; ends after 20 recovered steps.
* Minor spike (don't care): >= 6 sigma and >= 2%, not overlapping a major one.
* Early-warning horizon H = 200 steps; grace after episode end = 200 steps.
* An alarm in [onset - H, end] is a detection; lead = onset - alarm. Detection windows never reach back into the previous episode's [.., end + grace].
* False alarm: alarm outside every major/minor window. Window FA rate = fraction of disjoint, fully stable 1000-step windows with >= 1 false alarm.

## Primary endpoint

On each held-out series, at the G-rate operating point:
"detects N of M major spikes, N_ahead of them >= 1 step before onset (median lead K steps), at window false-alarm rate X (nominal <= 5%)".
**Success** = X <= 5% on every held-out OLMo series *and* N_ahead / M >= 50% pooled over held-out OLMo series.
**Falsified** if X > 5% on any held-out OLMo series (the guarantee failed to transfer) — reported as such, not re-tuned.

## Baselines (tuned, no guarantee)

* Robust z-threshold, alarm when max over signals z >= k (k grid 4..20); includes the OLMo-2 "7 sigma" spike rule.
* Page CUSUM on clipped z, drift 1, h grid.
* Each tuned on the calibration split to the most sensitive setting with window FA <= 5%, then applied unchanged to held-out series.

## Ablations (reported, not used for the endpoint)

rho = 1 vs stateful; each signal alone; optimizer-only signals (no loss/clip); G-run vs G-rate.

## Deviations

* **v1.1 (2026-11-18, before any held-out series was scored).** Signals are averaged over those
  *observed* at each step rather than all signals with e = 1 for unlogged ones. Motivation: at 9 of 10
  steps only loss/clip are logged and the 5 layer signals diluted their evidence 3.5x. Validity is
  unchanged (the logging schedule is predictable). Found by reading the smoke-test output on the
  calibration split; no held-out data was involved.
* **v1.2 (2026-11-18, before any small-run result was scored).** Own-runs arm fully specified. The
  first sweep (pre-LN, clipped at 1.0) and two pilots (pre-LN unclipped, lr up to 6e-2) produced *no*
  major spikes, so the arm uses Post-LN, no clipping, lr in {4,6,8,10}e-3, beta2 in {0.95, 0.999},
  seeds 0-2, 3000 steps. Signals: loss, gnorm, gmax, uratio, stale, curv (exact definitions,
  every step; clip is constant without clipping and is dropped); logitmax reported as an extra ablation.
  Residual window 100, conformal window 500, armed from step 1000. rho(s) fitted on the five lr = 1e-3
  control runs only. Same labels, H, grace, W, alpha as the OLMo arm.
