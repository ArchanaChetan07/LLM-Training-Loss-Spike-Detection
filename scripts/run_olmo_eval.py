"""Run the pre-registered OLMo evaluation.

    python scripts/run_olmo_eval.py              # all series present on disk
    python scripts/run_olmo_eval.py --calibrate  # (re)fit rho on OLMo-1B calibration split

Writes results/olmo_eval.json, results/calibration.json, results/olmo_alarms/*.npz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spikewatch.audit import fit_stateful  # noqa: E402
from spikewatch.experiment import (ALPHA_RUN, ALPHA_W, OPTIM_ONLY, SIGNALS, W, baseline_eval,  # noqa: E402
                                   cal_from_json, cal_to_json, clean, prepare, run_detector, tune_baseline)
from spikewatch.olmo import load_series, run_dirs  # noqa: E402

CAL_END = 250_000
BURN_IN = 10_000
RES = ROOT / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", nargs="*", default=["OLMo-1B", "OLMo-7B", "OLMo-1.7-7B", "OLMo-2-7B"])
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args()
    RES.mkdir(exist_ok=True); (RES / "olmo_alarms").mkdir(exist_ok=True)

    t0 = time.time()
    p1 = prepare("OLMo-1B", load_series("OLMo-1B", cache=not a.no_cache), BURN_IN)
    steps = p1.df["step"].to_numpy()
    # the calibration split is defined in optimizer steps; positions after the
    # first time the series passes CAL_END are test (rewinds stay in test)
    first_test = int(np.argmax(steps > CAL_END)) if (steps > CAL_END).any() else len(steps)
    cal_m = np.arange(len(steps)) < first_test
    print(f"OLMo-1B prepared {len(steps)} steps in {time.time()-t0:.0f}s; calibration positions < {first_test}")

    cpath = RES / "calibration.json"
    if a.calibrate or not cpath.exists():
        cals, table = fit_stateful(p1.sc.pvals, p1.mask_stable() & cal_m)
        cpath.write_text(json.dumps({"fitted_on": f"OLMo-1B steps {BURN_IN}..{CAL_END} stable",
                                     "calibrators": cal_to_json(cals), "table": clean(table)}, indent=1))
    cals = cal_from_json(json.loads(cpath.read_text())["calibrators"])

    # tune baselines on calibration split
    zk = tune_baseline(p1, "z", cal_m)
    ch = tune_baseline(p1, "cusum", cal_m)
    print(f"tuned baselines on calibration: z >= {zk}, CUSUM h = {ch}")

    out = {"protocol": "docs/PREREGISTRATION.md", "tuned": {"z": zk, "cusum": ch}, "series": {}}
    preps = {"OLMo-1B": p1}
    for s in a.series:
        if s not in preps:
            if not run_dirs(s):
                print(f"skip {s}: no data"); continue
            preps[s] = prepare(s, load_series(s, cache=not a.no_cache), BURN_IN)
    for s, p in preps.items():
        n = len(p.df)
        parts = {"calibration": cal_m, "test": ~cal_m} if s == "OLMo-1B" else {"test": np.ones(n, bool)}
        res = {"n_steps": n, "n_major": len(p.major), "n_minor": len(p.minor),
               "steps_range": [int(p.df.step.min()), int(p.df.step.max())]}
        for part, m in parts.items():
            T = int((p.armed & m).sum())
            r = {
                "G-rate": run_detector(p, cals, W / ALPHA_W, m=m),
                "G-run": run_detector(p, cals, T / ALPHA_RUN, m=m),
                "rho1_G-rate": run_detector(p, None, W / ALPHA_W, m=m),
                "optim_only_G-rate": run_detector(p, cals, W / ALPHA_W, signals=OPTIM_ONLY, m=m),
                "zscore_tuned": baseline_eval(p, "z", zk, m),
                "zscore_7sigma": baseline_eval(p, "z", 7, m),
                "cusum_tuned": baseline_eval(p, "cusum", ch, m),
            }
            for sig in SIGNALS:
                if sig in p.sc.pvals:
                    r[f"only_{sig}"] = run_detector(p, cals, W / ALPHA_W, signals=[sig], m=m)
            np.savez_compressed(RES / "olmo_alarms" / f"{s}_{part}.npz",
                                alarms=r["G-rate"]["_alarms"], logR=r["G-rate"]["_logR"].astype(np.float32))
            res[part] = clean(r)
            g = r["G-rate"]
            print(f"{s:12s} {part:11s} M={g['events']:3d} det={g['detected']:3d} ahead={g['detected_ahead_of_onset']:3d} "
                  f"lead~{g['median_lead_onset']} wFA={g['window_fa_rate']:.3f} | z{zk}: wFA={r['zscore_tuned']['window_fa_rate']:.3f} "
                  f"ahead={r['zscore_tuned']['detected_ahead_of_onset']} | rho1 wFA={r['rho1_G-rate']['window_fa_rate']:.3f}",
                  flush=True)
        res["episodes"] = [e.__dict__ for e in p.major]
        res["minor_episodes"] = [e.__dict__ for e in p.minor]
        out["series"][s] = clean(res)
    (RES / "olmo_eval.json").write_text(json.dumps(out, indent=1))
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
