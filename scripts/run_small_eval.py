"""Pre-registered evaluation on the small-model instability sweep (PREREGISTRATION v1.2).

rho(s) and baseline thresholds are fitted on the lr = 1e-3 control runs only;
every other run is test. Writes results/small_eval.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spikewatch.audit import fit_stateful  # noqa: E402
from spikewatch.evalues import SignalSpec  # noqa: E402
from spikewatch.experiment import (ALPHA_RUN, ALPHA_W, CUSUM_GRID, W, Z_GRID, baseline_eval,  # noqa: E402
                                   cal_to_json, clean, pool, prepare, run_detector)

SIGNALS = ["loss", "gnorm", "gmax", "uratio", "stale", "curv"]
EXTRA = ["logitmax"]
OPTIM = ["gnorm", "gmax", "uratio", "stale", "curv"]
SPECS = {s: SignalSpec(s, log=(s != "curv"), resid_window=100, conf_window=500) for s in SIGNALS + EXTRA}
BURN = 1000


def main():
    preps = {}
    for f in sorted((ROOT / "data" / "small").glob("*.parquet")):
        df = pd.read_parquet(f); df["run"] = f.stem
        preps[f.stem] = prepare(f.stem, df, BURN, signals=SIGNALS + EXTRA, specs=SPECS)
    ctrl = [k for k in preps if k.startswith("ctrl_")]
    test = [k for k in preps if not k.startswith("ctrl_")]
    print(f"{len(ctrl)} control runs, {len(test)} test runs; "
          f"major spikes in test: {sum(len(preps[k].major) for k in test)}, in controls: "
          f"{sum(len(preps[k].major) for k in ctrl)}")

    sigs = [s for s in SIGNALS + EXTRA if all(s in preps[k].sc.pvals for k in ctrl)]
    pv = {s: np.concatenate([preps[k].sc.pvals[s] for k in ctrl]) for s in sigs}
    cals, table = fit_stateful(pv, np.concatenate([preps[k].mask_stable() for k in ctrl]))

    def det(keys, cal, mode, signals):
        rs = []
        for k in keys:
            p = preps[k]
            thr = W / ALPHA_W if mode == "rate" else int(p.armed.sum()) / ALPHA_RUN
            rs.append(run_detector(p, cal, thr, signals=signals))
        return pool(rs)

    configs = {"G-rate": (cals, "rate", SIGNALS), "G-run": (cals, "run", SIGNALS),
               "rho1_G-rate": (None, "rate", SIGNALS), "optim_only_G-rate": (cals, "rate", OPTIM),
               "with_logitmax_G-rate": (cals, "rate", SIGNALS + EXTRA)}
    configs |= {f"only_{s}": (cals, "rate", [s]) for s in SIGNALS + EXTRA}
    out = {"calibration": cal_to_json(cals), "calibration_table": table, "controls": ctrl, "test_runs": test,
           "configs": {}}
    for name, (cal, mode, signals) in configs.items():
        out["configs"][name] = {"controls": det(ctrl, cal, mode, signals), "test": det(test, cal, mode, signals)}

    for kind, grid in (("z", Z_GRID), ("cusum", CUSUM_GRID)):
        bpool = lambda g, keys: pool([baseline_eval(preps[k], kind, g, signals=SIGNALS) for k in keys])
        best = grid[-1]
        for g in sorted(grid, reverse=True):  # most sensitive setting with control FA <= alpha
            if bpool(g, ctrl)["window_fa_rate"] <= ALPHA_W:
                best = g
            else:
                break
        out["configs"][f"{kind}_tuned"] = {"param": best, "controls": bpool(best, ctrl), "test": bpool(best, test)}
    out["configs"]["zscore_7sigma"] = {"param": 7, "controls": pool([baseline_eval(preps[k], "z", 7, signals=SIGNALS) for k in ctrl]),
                                       "test": pool([baseline_eval(preps[k], "z", 7, signals=SIGNALS) for k in test])}
    out["episodes"] = {k: [e.__dict__ for e in preps[k].major] for k in preps}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "small_eval.json").write_text(json.dumps(clean(out), indent=1))
    for name in ["G-rate", "G-run", "rho1_G-rate", "optim_only_G-rate", "z_tuned", "cusum_tuned", "zscore_7sigma"]:
        t, c = out["configs"][name]["test"], out["configs"][name]["controls"]
        print(f"{name:18s} test M={t['events']} det={t['detected']} ahead={t['detected_ahead_of_onset']} "
              f"lead~{t['median_lead_onset']} wFA={t['window_fa_rate']:.3f} | controls wFA={c['window_fa_rate']:.3f}")


if __name__ == "__main__":
    main()
