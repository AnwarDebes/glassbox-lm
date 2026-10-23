#!/usr/bin/env python
"""Summarise registered router runs: accuracy per arm, paired bootstrap of differences, composites,
calibration and out-of-scope AUROC. Reads results/confirm/<registration>/<name>/.

    report_router.py --registration R4 --tm <name> [--tm2 <name>] --lr <name> [--composite]

Accuracy differences: per test query, the mean correctness over seeds of each arm; paired bootstrap
over test queries (10,000 resamples, seed 0), 95% percentile interval.
Composites: two disjoint composites (seeds 1-5 and 6-10): sum of the stored normalised class sums.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def boot_diff(a: np.ndarray, b: np.ndarray, n: int = 10000, seed: int = 0) -> dict:
    """a, b: per-query mean correctness of two arms (same queries)."""
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, len(d), size=(n, len(d)))
    bs = d[idx].mean(1)
    return {"diff": float(d.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))}


def auroc(is_pos, score):
    from scipy.stats import rankdata

    is_pos = np.asarray(is_pos).astype(bool)
    r = rankdata(score)
    n1 = is_pos.sum()
    n0 = len(is_pos) - n1
    return float((r[is_pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def load_tm(folder: Path):
    seeds = sorted(int(p.stem[4:]) for p in folder.glob("seed*.json"))
    recs = {k: json.loads((folder / f"seed{k}.json").read_text()) for k in seeds}
    preds = {k: np.load(folder / f"preds_seed{k}.npz") for k in seeds}
    return seeds, recs, preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registration", required=True)
    ap.add_argument("--tm", required=True, help="run name of the TM arm")
    ap.add_argument("--tm2", default=None, help="optional second TM arm (compared with --tm)")
    ap.add_argument("--lr", default=None, help="run name holding lr.json (default: the TM arm's folder)")
    ap.add_argument("--composite", action="store_true")
    args = ap.parse_args()
    base = ROOT / "results" / "confirm" / args.registration
    out = {}
    seeds, recs, preds = load_tm(base / args.tm)
    y = preds[seeds[0]]["y"]
    corr = np.stack([preds[k]["pred"] == y for k in seeds]).astype(float)  # [seeds, n]
    accs = corr.mean(1)
    out["tm"] = {"name": args.tm, "seeds": seeds, "test_acc_mean": float(accs.mean()), "test_acc_sd": float(accs.std(ddof=1)) if len(seeds) > 1 else 0.0,
                 "test_acc_per_seed": [float(a) for a in accs],
                 "ece_test_mean": float(np.mean([recs[k]["calib"]["ece_test"] for k in seeds])),
                 "ece_test_max": float(np.max([recs[k]["calib"]["ece_test"] for k in seeds])),
                 "nll_test_mean": float(np.mean([recs[k]["calib"]["nll_test"] for k in seeds])),
                 "stats_first_member": [recs[k]["members"][0]["stats"] for k in seeds]}
    if "oos_auroc_test_maxsum" in recs[seeds[0]]:
        out["tm"]["oos_auroc_maxsum_per_seed"] = [recs[k]["oos_auroc_test_maxsum"] for k in seeds]
        out["tm"]["oos_auroc_maxsum_mean"] = float(np.mean(out["tm"]["oos_auroc_maxsum_per_seed"]))
        out["tm"]["oos_auroc_margin_mean"] = float(np.mean([recs[k]["oos_auroc_test_margin"] for k in seeds]))
    lr_dir = base / (args.lr or args.tm)
    if (lr_dir / "lr.json").exists():
        lr = json.loads((lr_dir / "lr.json").read_text())
        lrp = np.load(lr_dir / "lr_preds.npz")
        lr_corr = (lrp["pred"] == lrp["y"]).astype(float)
        assert np.array_equal(lrp["y"], y)
        out["lr"] = {"name": lr_dir.name, "test_acc": float(lr_corr.mean()), "C": lr["C"], "input": lr.get("input", "raw"),
                     "ece_test": lr["calib"]["ece_test"], "nll_test": lr["calib"]["nll_test"],
                     "oos_auroc_msp": lr.get("oos_auroc_test_msp")}
        out["tm_minus_lr"] = boot_diff(corr.mean(0), lr_corr)
    if args.tm2:
        s2, r2, p2 = load_tm(base / args.tm2)
        corr2 = np.stack([p2[k]["pred"] == y for k in s2]).astype(float)
        out["tm2"] = {"name": args.tm2, "seeds": s2, "test_acc_mean": float(corr2.mean(1).mean()),
                      "test_acc_per_seed": [float(a) for a in corr2.mean(1)],
                      "stats_first_member": [r2[k]["members"][0]["stats"] for k in s2]}
        out["tm2_minus_tm"] = boot_diff(corr2.mean(0), corr.mean(0))
        if "lr" in out:
            out["tm2_minus_lr"] = boot_diff(corr2.mean(0), lr_corr)
    if args.composite and len(seeds) >= 10:
        comps = []
        for grp in (seeds[:5], seeds[5:10]):
            S = sum(preds[k]["S_test"].astype(np.float32) for k in grp)
            comps.append(S.argmax(1) == y)
        cc = np.stack(comps).astype(float)
        out["composite"] = {"groups": [seeds[:5], seeds[5:10]], "test_acc_per_composite": [float(a) for a in cc.mean(1)],
                            "test_acc_mean": float(cc.mean())}
        if "lr" in out:
            out["composite_minus_lr"] = boot_diff(cc.mean(0), lr_corr)
        if "S_oos" in preds[seeds[0]]:
            au = []
            for grp in (seeds[:5], seeds[5:10]):
                Si = sum(preds[k]["S_test"].astype(np.float32) for k in grp).max(1)
                So = sum(preds[k]["S_oos"].astype(np.float32) for k in grp).max(1)
                au.append(auroc(np.r_[np.zeros(len(Si)), np.ones(len(So))], -np.r_[Si, So]))
            out["composite"]["oos_auroc_maxsum"] = au
    path = base / f"report_{args.tm}{('_vs_' + args.tm2) if args.tm2 else ''}.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("tm",)}, indent=1))
    print("TM", {k: v for k, v in out["tm"].items() if k != "stats_first_member"})


if __name__ == "__main__":
    main()
