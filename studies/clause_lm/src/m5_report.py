"""Aggregate the M5 step 1 runs and apply the pre-registered rules (docs/prereg_M5_step1.md).

Reads results/*_m5_*s<S>_seed<SEED>_<order>/metrics.json (written by src/m5_run.py),
prints a markdown table (s x seed), the per-s means, s* and the verdict, and logs
the aggregate as its own runlog folder (results/<UTC>_m5_summary_<order>/).

Rules (copied from the pre-registration, not re-decided here):
  s*      = the s with the highest mean VAL accuracy over the 3 seeds; ties -> lower s.
  SUCCESS = mean TEST accuracy at s* >= 0.848 AND every seed's test accuracy at s* > 0.828.
  NEGATIVE= s* = 8, or a gain below 2 points (mean test at s* - 0.828 < 0.02).
Usage: .venv-cdllm/bin/python src/m5_report.py [--label-order sorted] [--no-log]
"""
import argparse, glob, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runlog import ROOT, Run  # noqa: E402

S_GRID, SEEDS = [8.0, 25.0, 50.0, 100.0], [42, 123, 456]
BASE, SUCCESS_MEAN, SEED_FLOOR = 0.828, 0.848, 0.828


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-order", default="sorted")
    ap.add_argument("--no-log", action="store_true")
    a = ap.parse_args()

    runs, problems = {}, []
    pat = re.compile(rf"_m5_(?:gate0_)?s([0-9.]+)_seed(\d+)_{a.label_order}$")
    for d in sorted(glob.glob(f"{ROOT}/results/*_m5_*")):
        mt = pat.search(d)
        if not mt or not os.path.exists(f"{d}/metrics.json"):
            continue
        m = json.load(open(f"{d}/metrics.json"))
        if m.get("returncode") != 0 or "test_accuracy" not in m:
            problems.append(f"crashed/incomplete: {os.path.basename(d)}")
            continue
        key = (float(mt.group(1)), int(mt.group(2)))
        if key in runs:
            problems.append(f"DUPLICATE completed run for {key}: {os.path.basename(d)}")
            continue
        runs[key] = {**m, "folder": os.path.basename(d)}

    rows = ["| s | seed | val acc | test acc | OOS AUROC | non-empty clauses | median lits | mean lits | max lits | best epoch | train s | wall s | folder |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in S_GRID:
        for seed in SEEDS:
            r = runs.get((s, seed))
            if r is None:
                rows.append(f"| {s:g} | {seed} | missing | | | | | | | | | | |")
                continue
            c = r["clause_stats"]
            rows.append(f"| {s:g} | {seed} | {r['val_accuracy']:.4f} | {r['test_accuracy']:.4f} | {r['oos_auroc']:.4f} | "
                        f"{c['non_empty_clauses']} | {c['median_literals']:g} | {c['mean_literals']:.1f} | {c['max_literals']} | "
                        f"{r.get('best_epoch')} | {r['train_seconds']:.0f} | {r['wall_s']:.0f} | {r['folder']} |")

    per_s = {}
    for s in S_GRID:
        rs = [runs[(s, sd)] for sd in SEEDS if (s, sd) in runs]
        if len(rs) == len(SEEDS):
            mean = lambda k: sum(r[k] for r in rs) / len(rs)
            per_s[s] = {"mean_val": mean("val_accuracy"), "mean_test": mean("test_accuracy"),
                        "mean_oos_auroc": mean("oos_auroc"), "test_by_seed": [r["test_accuracy"] for r in rs],
                        "val_by_seed": [r["val_accuracy"] for r in rs],
                        "mean_median_lits": sum(r["clause_stats"]["median_literals"] for r in rs) / 3,
                        "mean_max_lits": sum(r["clause_stats"]["max_literals"] for r in rs) / 3,
                        "mean_wall_s": mean("wall_s")}
    means = ["| s | mean val | mean test | test by seed (42, 123, 456) | mean OOS AUROC | mean median lits | mean max lits | mean wall s |",
             "|---|---|---|---|---|---|---|---|"]
    for s, p in per_s.items():
        means.append(f"| {s:g} | {p['mean_val']:.4f} | {p['mean_test']:.4f} | {', '.join(f'{x:.4f}' for x in p['test_by_seed'])} | "
                     f"{p['mean_oos_auroc']:.4f} | {p['mean_median_lits']:g} | {p['mean_max_lits']:.0f} | {p['mean_wall_s']:.0f} |")

    out = {"label_order": a.label_order, "n_completed_runs": len(runs), "problems": problems,
           "per_s": {f"{s:g}": p for s, p in per_s.items()}}
    if len(per_s) == len(S_GRID):
        best_val = max(p["mean_val"] for p in per_s.values())
        s_star = min(s for s, p in per_s.items() if p["mean_val"] == best_val)  # ties -> lower s
        p = per_s[s_star]
        gain = p["mean_test"] - BASE
        success = p["mean_test"] >= SUCCESS_MEAN and all(t > SEED_FLOOR for t in p["test_by_seed"])
        negative = (s_star == 8.0) or (gain < 0.02)
        verdict = "SUCCESS" if success else ("NEGATIVE" if negative else "NOT SUCCESS, NOT the pre-registered negative case")
        out.update({"s_star": s_star, "s_star_mean_val": p["mean_val"], "s_star_mean_test": p["mean_test"],
                    "s_star_test_by_seed": p["test_by_seed"], "gain_vs_0.828_points": 100 * gain,
                    "success_rule_met": success, "negative_rule_met": negative, "verdict": verdict,
                    "ties_at_best_val": [f"{s:g}" for s, q in per_s.items() if q["mean_val"] == best_val]})
    text = "\n".join(rows) + "\n\n" + "\n".join(means) + "\n\n" + json.dumps(
        {k: v for k, v in out.items() if k != "per_s"}, indent=1)
    print(text)
    if not a.no_log:
        run = Run(f"m5_summary_{a.label_order}", {"inputs": sorted(r["folder"] for r in runs.values()),
                                                   "rules": __doc__}, script=__file__)
        open(f"{run.dir}/tables.md", "w").write(text + "\n")
        run.finish(out)


if __name__ == "__main__":
    main()
