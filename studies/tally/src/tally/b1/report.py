"""B1 verdict from the run folders.

    PYTHONPATH=src python -m tally.b1.report --label reg [--out results/b1_report_reg.json]

Reads every finished run b1_<label>_<arm>_M<M>_s<seed> (the latest per key), and computes the
registered statistics. Censored runs (target never reached) enter at the cap.

Primary: at M = 128, ratio of means shared / marginal over the seeds, 95% interval from a paired
bootstrap over seeds (B_BOOT resamples of the seed indices, percentile interval).
    success: ratio >= 5, lower bound > 2, and no marginal run at M = 128 censored
    kill:    ratio < 2, upper bound < 5, and no shared run at M = 128 censored
    otherwise no claim (the reason is printed).
Second outcome: slope of log ratio against log2 M over the sizes where both arms have runs, with
the same seed resamples at every size; "grows" when the 95% interval of the slope is above 0.
Dual-state trial: reduction = 1 - mean(dual) / mean(marginal, direct) at M = 128, paired
bootstrap interval.
    success: reduction >= 0.25, never ends worse (at no M is the paired mean final excess of
             dual minus direct above 0 with its 95% interval above 0), and no dual run at M = 128
             censored
    kill:    reduction < 0.05 and no marginal run at M = 128 censored
    otherwise no claim (the reason is printed).
A censored dual run enters below its true count, which biases the reduction upward; a censored
marginal run biases it downward. Changed 2026-10-25: before, the dual line had no censoring
condition, so dual and marginal runs all censored at the cap gave a reduction of 0, read "kill".
The fixed-sample A/B arm is reported against both.
"""
import argparse
import glob
import json
import os
import re

import numpy as np

SIZES = [8, 16, 32, 64, 128, 256, 512]
SEEDS = list(range(10))
B_BOOT = 10_000
PRIMARY_M = 128
SUCCESS_X, KILL_X = 5.0, 2.0
DUAL_SUCCESS, DUAL_KILL = 0.25, 0.05


def load(label):
    from tally.runlog import RESULTS
    pat = re.compile(rf"^\d{{8}}T\d{{6}}Z_b1_{re.escape(label)}_([a-z_]+)_M(\d+)_s(\d+)$")
    runs = {}
    for d in sorted(glob.glob(os.path.join(RESULTS, f"*_b1_{label}_*"))):
        m = pat.match(os.path.basename(d))
        if not m:
            continue
        try:
            with open(os.path.join(d, "run.json")) as f:
                if json.load(f).get("state") != "finished":
                    continue
            with open(os.path.join(d, "metrics.json")) as f:
                met = json.load(f)
        except OSError:
            continue
        runs[(m.group(1), int(m.group(2)), int(m.group(3)))] = (d, met)   # latest wins
    return runs


def table(runs, arm, M, key):
    return np.array([runs[(arm, M, s)][1][key] if (arm, M, s) in runs else np.nan for s in SEEDS],
                    dtype=float)


def boot_idx(n, rng):
    return rng.integers(0, n, size=(B_BOOT, n))


def ratio_ci(a, b, idx):
    """mean(a) / mean(b) with percentile interval over resampled seed indices."""
    est = a.mean() / b.mean()
    bs = a[idx].mean(1) / b[idx].mean(1)
    return float(est), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def analyse(runs, label):
    rng = np.random.default_rng(12345)
    idx = boot_idx(len(SEEDS), rng)
    out = {"label": label, "n_runs": len(runs), "summary": {}, "missing": []}
    arms = sorted({k[0] for k in runs})
    for arm in arms:
        for M in SIZES:
            st = table(runs, arm, M, "samples_to_target")
            if np.all(np.isnan(st)):
                continue
            cen = table(runs, arm, M, "censored")
            fe = table(runs, arm, M, "final_excess")
            out["summary"][f"{arm}_M{M}"] = {
                "n": int(np.sum(~np.isnan(st))), "mean_samples": float(np.nanmean(st)),
                "censored": int(np.nansum(cen)), "mean_final_excess": float(np.nanmean(fe))}
    for arm in ("shared", "selfish", "marginal", "dual", "ab", "greedy", "random", "flat", "oracle"):
        for M in SIZES:
            for s in SEEDS:
                if (arm, M, s) not in runs:
                    out["missing"].append(f"{arm}_M{M}_s{s}")

    def complete(arm, M):
        return all((arm, M, s) in runs for s in SEEDS)

    # primary
    v = {}
    if complete("shared", PRIMARY_M) and complete("marginal", PRIMARY_M):
        sh = table(runs, "shared", PRIMARY_M, "samples_to_target")
        mg = table(runs, "marginal", PRIMARY_M, "samples_to_target")
        r, ci = ratio_ci(sh, mg, idx)
        cen_sh = int(table(runs, "shared", PRIMARY_M, "censored").sum())
        cen_mg = int(table(runs, "marginal", PRIMARY_M, "censored").sum())
        if r >= SUCCESS_X and ci[0] > KILL_X and cen_mg == 0:
            verdict = "success"
        elif r < KILL_X and ci[1] < SUCCESS_X and cen_sh == 0:
            verdict = "kill"
        else:
            verdict = "no claim"
        why = []
        if cen_mg:
            why.append(f"{cen_mg} marginal runs censored (ratio biased upward; success blocked)")
        if cen_sh:
            why.append(f"{cen_sh} shared runs censored (ratio biased downward; kill blocked)")
        v["primary"] = {"M": PRIMARY_M, "ratio_shared_over_marginal": r, "ci95": ci,
                        "censored_shared": cen_sh, "censored_marginal": cen_mg,
                        "verdict": verdict, "notes": why}
    else:
        v["primary"] = {"verdict": "incomplete"}
    # second outcome: slope of log ratio over log2 M
    Ms = [M for M in SIZES if complete("shared", M) and complete("marginal", M)]
    if len(Ms) >= 3:
        S = np.array([table(runs, "shared", M, "samples_to_target") for M in Ms])
        G = np.array([table(runs, "marginal", M, "samples_to_target") for M in Ms])
        xs = np.log2(Ms)

        def slope(Si, Gi):
            return float(np.polyfit(xs, np.log(Si.mean(1) / Gi.mean(1)), 1)[0])
        est = slope(S, G)
        bs = np.array([slope(S[:, ii], G[:, ii]) for ii in idx])
        ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        v["slope"] = {"sizes": Ms, "slope_log_ratio_per_doubling": est, "ci95": ci,
                      "ratios": [float(S[j].mean() / G[j].mean()) for j in range(len(Ms))],
                      "grows": bool(ci[0] > 0)}
    # dual-state trial
    if complete("dual", PRIMARY_M) and complete("marginal", PRIMARY_M):
        du = table(runs, "dual", PRIMARY_M, "samples_to_target")
        mg = table(runs, "marginal", PRIMARY_M, "samples_to_target")
        r, ci = ratio_ci(du, mg, idx)
        worse = []
        for M in SIZES:
            if complete("dual", M) and complete("marginal", M):
                d = table(runs, "dual", M, "final_excess") - table(runs, "marginal", M, "final_excess")
                bs = d[idx].mean(1)
                lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
                if d.mean() > 0 and lo > 0:
                    worse.append({"M": M, "mean_diff": float(d.mean()), "ci95": [lo, hi]})
        red = 1 - r
        red_ci = [1 - ci[1], 1 - ci[0]]
        cen = int(table(runs, "dual", PRIMARY_M, "censored").sum())
        cen_mg = int(table(runs, "marginal", PRIMARY_M, "censored").sum())
        if red >= DUAL_SUCCESS and not worse and cen == 0:
            dv = "success"
        elif red < DUAL_KILL and cen_mg == 0:
            dv = "kill"
        else:
            dv = "no claim"
        why = []
        if cen:
            why.append(f"{cen} dual runs censored (reduction biased upward; success blocked)")
        if cen_mg:
            why.append(f"{cen_mg} marginal runs censored (reduction biased downward; kill blocked)")
        v["dual"] = {"M": PRIMARY_M, "reduction": red, "ci95": red_ci, "ends_worse_at": worse,
                     "censored_dual": cen, "censored_marginal": cen_mg, "verdict": dv, "notes": why}
        if complete("ab", PRIMARY_M):
            ab = table(runs, "ab", PRIMARY_M, "samples_to_target")
            v["ab"] = {"ab_over_direct": ratio_ci(ab, mg, idx), "ab_over_dual": ratio_ci(ab, du, idx)}
    out["verdict"] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="reg")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = analyse(load(a.label), a.label)
    v = out["verdict"]
    txt = json.dumps(out, indent=1, sort_keys=True)
    print(json.dumps(v, indent=1, sort_keys=True))
    print("\narm_M: n, mean samples to target, censored, mean final excess")
    for k, s in out["summary"].items():
        print(f"  {k:16s} {s['n']:3d} {s['mean_samples']:12.0f} {s['censored']:3d} {s['mean_final_excess']:.4f}")
    print(f"missing runs: {len(out['missing'])}")
    if a.out:
        with open(a.out, "w") as f:
            f.write(txt)


if __name__ == "__main__":
    main()
