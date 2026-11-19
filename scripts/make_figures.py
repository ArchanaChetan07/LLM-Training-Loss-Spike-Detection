"""Figures for the README / report. Reads data + results/calibration.json.

  fig1_case_study.png      one OLMo spike: loss, signal tail-ranks, detector statistic
  fig2_guarantee.png       empirical vs nominal window false-alarm rate (the guarantee, audited)
  fig3_tradeoff.png        early detection vs false alarms: e-detector vs tuned baselines
  fig4_transfer.png        false-alarm rate per held-out series at the tuned/nominal 5% point
  fig5_leadtime.png        lead-time distribution
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spikewatch.experiment import (ALPHA_W, CUSUM_GRID, OPTIM_ONLY, W, Z_GRID, baseline_eval,  # noqa: E402
                                   cal_from_json, pool, prepare, run_detector)
from spikewatch.olmo import load_series, run_dirs  # noqa: E402

FIG = ROOT / "figures"
C = {"av": "#2a78d6", "rho1": "#6250d6", "z": "#eb6834", "cusum": "#1baf7a", "ink": "#0b0b0b",
     "muted": "#52514e", "grid": "#e4e3df", "bound": "#8a8986"}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": C["muted"], "axes.labelcolor": C["ink"],
                     "xtick.color": C["muted"], "ytick.color": C["muted"], "axes.grid": True,
                     "grid.color": C["grid"], "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight",
                     "lines.linewidth": 2})
BURN, CAL_END = 10_000, 250_000
ALPHAS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]


def held_out():
    out = {}
    for s in ["OLMo-1B", "OLMo-7B", "OLMo-1.7-7B", "OLMo-2-7B"]:
        if not run_dirs(s):
            continue
        p = prepare(s, load_series(s), BURN)
        steps = p.df.step.to_numpy()
        m = np.ones(len(steps), bool)
        if s == "OLMo-1B":
            m = np.arange(len(steps)) >= int(np.argmax(steps > CAL_END))
        out[s] = (p, m)
    return out


def fig_case(p, cals, ep):
    a, b = ep.onset - 400, ep.onset + 60
    sl = slice(a, b)
    r = run_detector(p, cals, W / ALPHA_W)
    x = p.df.step.to_numpy()[sl]
    fig, ax = plt.subplots(3, 1, figsize=(7.5, 6.2), sharex=True, gridspec_kw={"height_ratios": [1.2, 1.4, 1]})
    ax[0].plot(x, p.df.loss.to_numpy()[sl], color=C["ink"], lw=1.2)
    ax[0].set_ylabel("train loss")
    ax[0].set_title(f"{p.name} spike at step {ep.step}: +{100*ep.severity:.0f}% loss", loc="left", fontsize=11)
    for k, (sname, col) in enumerate([("gnorm", C["av"]), ("uratio", C["z"]), ("stale", C["cusum"]),
                                       ("loss", C["rho1"])]):
        if sname not in p.sc.pvals:
            continue
        pv = p.sc.pvals[sname][sl]; ok = np.isfinite(pv)
        ax[1].plot(x[ok], -np.log10(pv[ok]), "o-", ms=3, lw=1, color=col, label=sname)
    ax[1].axhline(2, color=C["bound"], lw=1, ls="--")
    ax[1].text(x[0], 2.08, "top 1% of own history", color=C["muted"], fontsize=8)
    ax[1].set_ylabel("tail rank  −log10 p")
    ax[1].legend(ncol=4, frameon=False, fontsize=8, loc="upper left")
    ax[2].plot(x, r["_logR"][sl] / np.log(10), color=C["av"])
    ax[2].axhline(np.log10(W / ALPHA_W), color=C["bound"], ls="--", lw=1)
    ax[2].text(x[0], np.log10(W / ALPHA_W) + 0.1, "alarm threshold (≤5% false-alarm windows)", color=C["muted"], fontsize=8)
    ax[2].set_ylabel("log10 R (e-detector)")
    al = r["_alarms"]; al = al[(al >= a) & (al < b)]
    for axx in ax:
        axx.axvline(ep.step, color=C["muted"], lw=1, ls=":")
        if al.size:
            axx.axvline(p.df.step.iloc[al[0]], color=C["av"], lw=1)
    ax[2].set_xlabel("optimizer step")
    fig.savefig(FIG / "fig1_case_study.png"); plt.close(fig)


def fig_guarantee(ho, cals):
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.plot([0, 0.2], [0, 0.2], color=C["bound"], ls="--", lw=1)
    ax.text(0.12, 0.13, "guarantee: empirical ≤ nominal", rotation=38, color=C["muted"], fontsize=8)
    marks = ["o", "s", "^", "D"]
    for (s, (p, m)), mk in zip(ho.items(), marks):
        for cal, col, lab in ((cals, C["av"], "cluster-aware null"), (None, C["rho1"], "exchangeable null (ρ=1)")):
            ys = [run_detector(p, cal, W / a, m=m)["window_fa_rate"] for a in ALPHAS]
            ax.plot(ALPHAS, ys, marker=mk, ms=6, lw=1.2, color=col, label=f"{s} · {lab}")
    ax.set_xlabel("nominal false-alarm rate per 1000-step window (α)")
    ax.set_ylabel("empirical rate on held-out stable windows")
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(FIG / "fig2_guarantee.png"); plt.close(fig)


def fig_tradeoff(ho, cals):
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    def curve(fn, grid):
        pts = []
        for g in grid:
            pr = pool([fn(p, m, g) for p, m in ho.values()])
            pts.append((pr["window_fa_rate"], pr["detected_ahead_of_onset"] / max(pr["events"], 1)))
        return np.array(pts)
    av = curve(lambda p, m, a: run_detector(p, cals, W / a, m=m), ALPHAS)
    z = curve(lambda p, m, k: baseline_eval(p, "z", k, m), Z_GRID)
    cu = curve(lambda p, m, h: baseline_eval(p, "cusum", h, m), CUSUM_GRID)
    for pts, col, lab in ((av, C["av"], "e-detector (α sweep)"), (z, C["z"], "max robust-z (k sweep)"),
                          (cu, C["cusum"], "CUSUM (h sweep)")):
        o = np.argsort(pts[:, 0])
        ax.plot(pts[o, 0], pts[o, 1], "o-", ms=6, color=col, label=lab)
    ax.axvline(0.05, color=C["bound"], ls="--", lw=1)
    ax.set_xlabel("false-alarm rate per 1000-step stable window (held-out)")
    ax.set_ylabel("fraction of spikes flagged ≥1 step before onset")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(FIG / "fig3_tradeoff.png"); plt.close(fig)
    return av, z, cu


def fig_transfer(res):
    series = [s for s in res["series"]]
    rows = []
    for s in series:
        part = res["series"][s]["test"]
        rows.append((s, part["G-rate"]["window_fa_rate"], part["zscore_tuned"]["window_fa_rate"],
                     part["cusum_tuned"]["window_fa_rate"], part["zscore_7sigma"]["window_fa_rate"]))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    xs = np.arange(len(rows)); wbar = 0.2
    for j, (lab, col) in enumerate([("e-detector (no tuning)", C["av"]), ("robust-z, tuned on OLMo-1B", C["z"]),
                                     ("CUSUM, tuned on OLMo-1B", C["cusum"]), ("7σ rule", C["rho1"])]):
        v = [r[j + 1] or 0 for r in rows]
        ax.bar(xs + (j - 1.5) * wbar, v, wbar * 0.9, color=col, label=lab)
    ax.axhline(0.05, color=C["bound"], ls="--", lw=1); ax.text(-0.45, 0.053, "5% target", color=C["muted"], fontsize=8)
    ax.set_xticks(xs, [r[0] + ("\n(held-out steps)" if r[0] == "OLMo-1B" else "") for r in rows])
    ax.set_ylabel("false-alarm windows")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(FIG / "fig4_transfer.png"); plt.close(fig)


def fig_lead(res, small=None):
    leads = [x for s in res["series"].values() for x in s["test"]["G-rate"]["lead_onset"] if x is not None]
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    bins = np.arange(-10.5, max(30, max(leads, default=0)) + 1.5, 1)
    ax.hist(np.clip(leads, -10, 60), bins=bins, color=C["av"], label=f"OLMo held-out (n={len(leads)})", alpha=0.9)
    if small:
        sl = small["configs"]["G-rate"]["test"]["lead_onset"]
        ax.hist(np.clip(sl, -10, 60), bins=bins, color=C["z"], alpha=0.6, label=f"own runs (n={len(sl)})")
    ax.axvline(0.5, color=C["muted"], ls=":", lw=1)
    ax.set_xlabel("lead time to onset (steps; >0 = warned before the spike)")
    ax.set_ylabel("spikes")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(FIG / "fig5_leadtime.png"); plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    cals = cal_from_json(json.loads((ROOT / "results/calibration.json").read_text())["calibrators"])
    res = json.loads((ROOT / "results/olmo_eval.json").read_text())
    small = json.loads((ROOT / "results/small_eval.json").read_text()) if (ROOT / "results/small_eval.json").exists() else None
    ho = held_out()
    p, m = ho["OLMo-1B"]
    eps = [e for e in p.major if m[e.onset]] or p.major
    ep = max(eps, key=lambda e: e.severity)
    fig_case(p, cals, ep)
    fig_guarantee(ho, cals)
    av, z, cu = fig_tradeoff(ho, cals)
    json.dump({"av": av.tolist(), "z": z.tolist(), "cusum": cu.tolist(), "alphas": ALPHAS, "z_grid": Z_GRID,
               "cusum_grid": CUSUM_GRID}, open(ROOT / "results/tradeoff.json", "w"), indent=1)
    fig_transfer(res)
    fig_lead(res, small)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
