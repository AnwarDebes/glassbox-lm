#!/usr/bin/env python
"""M1, the verdict (tu/PLAN_M1.md "Hypotheses"): H1, H2, H3a, H3b and the descriptive lines, always for all 3 sets.

    python -I -B m1_verdict.py --m1 tu/m1 --overlap tu/m1/overlap_m1.json

Input: <m1>/results_m1_tm.jsonl (tag "m1", seeds 7-9, s 8), <m1>/results_m1_lr.jsonl, the test predictions they name,
the test labels, and the step-0 overlap file. Every number is computed from the saved predictions (each row's
test_acc must equal its predictions' accuracy rounded to 4 decimals). Points throughout.
- TM value of a configuration = the mean over seeds 7-9 of its test accuracy. LR = its one fit.
- H1, per set: a3B - C4 >= +0.5 with TM AND with LR.
- H2, per set: a3B - potion >= -0.5 with TM AND with LR ("within 0.5 points of potion-base-32M"; never "beats").
- H3a, per set: LR a3B_3c - LR a3B >= +0.5 (augmentation with 3c's paraphrases helps).
- H3b, per set: LR a3B_3c - LR a3B_twin >= +0.5 (the paraphrases beat the EDA copies of the same rows).
- A hypothesis is MET when it holds on the point estimates on at least 2 of 3 sets. "strong" on a set when the lower
  end of the 95% interval clears the same bar with every probe the hypothesis uses; STRONG when that holds on at least
  2 of 3 sets. The interval: a paired bootstrap over test examples, 10,000 resamples per set, numpy
  default_rng([20261021, set index]); one index draw per resample is shared by every comparison of that set and by
  seeds 7-9 (the TM statistic is the seed-averaged paired difference on the drawn examples); percentiles 2.5 / 97.5.
  LIMIT: the interval resamples test examples only and leaves out the variance between training seeds. So beside
  every TM interval the three per-seed differences and the paired SE over seeds are given, and every TM
  configuration's seed SD, so a reader can see when one seed carries a gain.
- The four hypotheses are separate claims with no correction for multiplicity: meeting one says nothing about another.
- Descriptive (no bar): TM a3B_3c - a3B (repeat 1, as registered on val), a3B_3c - a3B repeat 5 (matched updates),
  a3B repeat 5 - a3B, a3B_3c - a3B_twin, a3B_twin - a3B; the R2 ablation a3B_3c - C4_3c (TM and LR); LR a3B_twin - a3B;
  each with its interval and, for TM, the paired SE over seeds; every configuration's val mean beside its test mean;
  each arm's test accuracy without the step-0 matched test texts.
Refuses to write unless every configuration has exactly seeds 7, 8 and 9, one row each, s 8, no smoke row, and the
same TM settings apart from the seed; and every LR arm has exactly one row. Refuses to overwrite.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

SETS = ("clinc150", "banking77", "hwu64")
SEEDS = (7, 8, 9)
TM_CONFIGS = (("C4", 1), ("a3B", 1), ("potion", 1), ("a3B_3c", 1), ("C4_3c", 1), ("a3B_twin", 1), ("a3B", 5))
LR_ARMS = ("C4", "a3B", "potion", "a3B_3c", "C4_3c", "a3B_twin")
N_BOOT, BOOT_SEED, CHUNK = 10_000, 20261021, 250
EPS = 1e-9
# (name, probe, a, b, bar or None); a TM side is (arm, repeat)
COMPARISONS = (
    ("H1_tm", "tm", ("a3B", 1), ("C4", 1), 0.5), ("H1_lr", "lr", "a3B", "C4", 0.5),
    ("H2_tm", "tm", ("a3B", 1), ("potion", 1), -0.5), ("H2_lr", "lr", "a3B", "potion", -0.5),
    ("H3a_lr", "lr", "a3B_3c", "a3B", 0.5), ("H3b_lr", "lr", "a3B_3c", "a3B_twin", 0.5),
    ("tm_3c_minus_a3B", "tm", ("a3B_3c", 1), ("a3B", 1), None),
    ("tm_3c_minus_a3B_repeat5", "tm", ("a3B_3c", 1), ("a3B", 5), None),
    ("tm_a3B_repeat5_minus_a3B", "tm", ("a3B", 5), ("a3B", 1), None),
    ("tm_3c_minus_twin", "tm", ("a3B_3c", 1), ("a3B_twin", 1), None),
    ("tm_twin_minus_a3B", "tm", ("a3B_twin", 1), ("a3B", 1), None),
    ("tm_ablation_a3B_3c_minus_C4_3c", "tm", ("a3B_3c", 1), ("C4_3c", 1), None),
    ("lr_ablation_a3B_3c_minus_C4_3c", "lr", "a3B_3c", "C4_3c", None),
    ("lr_twin_minus_a3B", "lr", "a3B_twin", "a3B", None),
)
HYP = {"H1": ("H1_tm", "H1_lr"), "H2": ("H2_tm", "H2_lr"), "H3a": ("H3a_lr",), "H3b": ("H3b_lr",)}


def show(n: str, point: dict, ci: dict, se: dict, per: dict) -> str:
    """One comparison as text: the point, the interval and, for TM, the per-seed differences and the seed SE."""
    s = f"{n} {point[n]:+.2f} [{ci[n][0]:+.2f}, {ci[n][1]:+.2f}]"
    if n in se:
        s += f" (per seed {', '.join(f'{v:+.2f}' for v in per[n])}; seed SE {se[n]:.2f})"
    return s


def check_rows(tm: list[dict], lr: list[dict]) -> list[str]:
    errs = []
    if any(r.get("smoke") for r in tm):
        errs.append("a smoke row is in the TM results")
    rows = [r for r in tm if r.get("tag") == "m1"]
    cfg0 = None
    for d in SETS:
        for arm, rep in TM_CONFIGS:
            got = [r for r in rows if r["dataset"] == d and r["arm"] == arm and r["repeat"] == rep]
            seeds = sorted(r["seed"] for r in got)
            if seeds != list(SEEDS):
                errs.append(f"TM {d} {arm} repeat {rep}: seeds {seeds}, expected {list(SEEDS)}")
            for r in got:
                if r["s"] != 8.0:
                    errs.append(f"TM {d} {arm} repeat {rep} seed {r['seed']}: s {r['s']}")
                c = {k: v for k, v in r["cfg"].items() if k != "seed"}
                if cfg0 is None:
                    cfg0 = c
                elif c != cfg0:
                    errs.append(f"TM {d} {arm} repeat {rep} seed {r['seed']}: TM settings differ")
        for arm in LR_ARMS:
            n = sum(1 for r in lr if r["dataset"] == d and r["name"] == arm)
            if n != 1:
                errs.append(f"LR {d} {arm}: {n} rows, expected 1")
    return errs


def boot_intervals(D: dict[str, np.ndarray], set_index: int) -> dict[str, tuple[float, float]]:
    """95% percentile intervals (points) of mean(D[name][idx]) over N_BOOT shared index draws."""
    n = len(next(iter(D.values())))
    rng = np.random.default_rng([BOOT_SEED, set_index])
    stats = {k: [] for k in D}
    for _ in range(N_BOOT // CHUNK):
        idx = rng.integers(0, n, size=(CHUNK, n))
        for k, v in D.items():
            stats[k].append(v[idx].mean(1))
    out = {}
    for k, v in stats.items():
        s = 100 * np.concatenate(v)
        out[k] = (round(float(np.percentile(s, 2.5)), 4), round(float(np.percentile(s, 97.5)), 4))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--m1", type=Path, required=True)
    ap.add_argument("--overlap", type=Path, default=None)
    ap.add_argument("--labels", default="", help="tests only: FILE.py:function returning a stand-in dataset")
    a = ap.parse_args()
    outj, outt = a.m1 / "verdict_m1.json", a.m1 / "verdict_m1.txt"
    for p in (outj, outt):
        if p.exists():
            print(f"refusing to overwrite {p}", file=sys.stderr)
            return 2
    tm = [json.loads(x) for x in (a.m1 / "results_m1_tm.jsonl").read_text().splitlines() if x.strip()]
    lr = [json.loads(x) for x in (a.m1 / "results_m1_lr.jsonl").read_text().splitlines() if x.strip()]
    errs = check_rows(tm, lr)
    if errs:
        print("REFUSED:\n" + "\n".join(errs), file=sys.stderr)
        return 3
    ov = json.loads(a.overlap.read_text()) if a.overlap else None
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from m1_feats import load_ds

    T = {(d, arm, rep): {r["seed"]: r for r in tm if r.get("tag") == "m1" and r["dataset"] == d and r["arm"] == arm
                          and r["repeat"] == rep} for d in SETS for arm, rep in TM_CONFIGS}
    L = {(d, arm): next(r for r in lr if r["dataset"] == d and r["name"] == arm) for d in SETS for arm in LR_ARMS}
    res, lines = {"sets": {}, "hypotheses": {}, "bootstrap": {"resamples": N_BOOT, "seed": BOOT_SEED}}, []
    count = {h: [0, 0] for h in HYP}  # [met, strong]
    for si, d in enumerate(SETS):
        y = load_ds(d, a.labels).labels["test"]
        ok_tm, ok_lr = {}, {}
        for arm, rep in TM_CONFIGS:
            rows = T[(d, arm, rep)]
            m = np.stack([(np.load(rows[s]["preds"]) == y) for s in SEEDS]).astype(np.float64)
            for j, s in enumerate(SEEDS):
                if round(float(m[j].mean()), 4) != rows[s]["test_acc"]:
                    print(f"REFUSED: {d} {arm}@{rep} seed {s}: predictions give {m[j].mean():.4f}, row {rows[s]['test_acc']}",
                          file=sys.stderr)
                    return 4
            ok_tm[(arm, rep)] = m
        for arm in LR_ARMS:
            m = (np.load(L[(d, arm)]["preds"]) == y).astype(np.float64)
            if round(float(m.mean()), 4) != L[(d, arm)]["lr_test"]:
                print(f"REFUSED: {d} LR {arm}: predictions give {m.mean():.4f}, row {L[(d, arm)]['lr_test']}", file=sys.stderr)
                return 4
            ok_lr[arm] = m
        D, point, se, per = {}, {}, {}, {}
        for name, probe, x, z, _bar in COMPARISONS:
            if probe == "tm":
                diff = ok_tm[x] - ok_tm[z]  # seeds x examples
                D[name] = diff.mean(0)
                per_seed = 100 * diff.mean(1)
                per[name] = [round(float(v), 4) for v in per_seed]
                se[name] = round(float(per_seed.std(ddof=1) / math.sqrt(len(SEEDS))), 4)
            else:
                D[name] = ok_lr[x] - ok_lr[z]
            point[name] = round(float(100 * D[name].mean()), 4)
        ci = boot_intervals(D, si)
        comp = {name: {"point": point[name], "ci95": ci[name],
                       **({"per_seed": per[name], "seed_se": se[name]} if name in se else {}),
                       **({"bar": bar} if bar is not None else {})} for name, _p, _x, _z, bar in COMPARISONS}
        hyp = {}
        for h, parts in HYP.items():
            met = all(point[p] >= comp[p]["bar"] - EPS for p in parts)
            strong = all(ci[p][0] >= comp[p]["bar"] - EPS for p in parts)
            hyp[h] = {"met": met, "strong": strong}
            count[h][0] += met
            count[h][1] += strong
        means = {f"{arm}@{rep}": {"test": round(float(100 * ok_tm[(arm, rep)].mean()), 4),
                                  "test_per_seed": [round(float(100 * v), 4) for v in ok_tm[(arm, rep)].mean(1)],
                                  "test_seed_sd": round(float(100 * ok_tm[(arm, rep)].mean(1).std(ddof=1)), 4),
                                  "val": round(100 * sum(T[(d, arm, rep)][s]["val_acc"] for s in SEEDS) / len(SEEDS), 4)}
                 for arm, rep in TM_CONFIGS}
        lrm = {arm: {"test": round(float(100 * ok_lr[arm].mean()), 4), "val": round(100 * L[(d, arm)]["lr_val"], 4),
                     "C": L[(d, arm)]["C"]} for arm in LR_ARMS}
        without = None
        if ov is not None:
            keep = np.ones(len(y), bool)
            keep[ov["sets"][d]["matched_test_idx"]] = False
            without = {"n_dropped": int((~keep).sum()),
                       "tm": {f"{arm}@{rep}": round(float(100 * ok_tm[(arm, rep)][:, keep].mean()), 4) for arm, rep in TM_CONFIGS},
                       "lr": {arm: round(float(100 * ok_lr[arm][keep].mean()), 4) for arm in LR_ARMS}}
        res["sets"][d] = {"n_test": len(y), "hypotheses": hyp, "comparisons": comp, "tm": means, "lr": lrm,
                          "without_matched_test_texts": without}
        lines.append(f"{d} (n_test {len(y)}): TM " + " / ".join(f"{k} {v['test']:.2f} (seed SD {v['test_seed_sd']:.2f})"
                                                              for k, v in means.items())
                     + "; LR " + " / ".join(f"{k} {v['test']:.2f}" for k, v in lrm.items()))
        for h, parts in HYP.items():
            lines.append(f"  {h} {'meets' if hyp[h]['met'] else 'misses'}{' (strong)' if hyp[h]['strong'] else ''}: "
                         + "; ".join(show(p, point, ci, se, per) for p in parts))
        lines.append("  descriptive: " + "; ".join(show(n, point, ci, se, per) for n, _p, _x, _z, bar in COMPARISONS
                                                  if bar is None))
    for h in HYP:
        res["hypotheses"][h] = {"sets_met": count[h][0], "met": count[h][0] >= 2, "sets_strong": count[h][1],
                                "strong": count[h][1] >= 2}
    head = "VERDICT M1 (test): " + "; ".join(
        f"{h} {'MET' if count[h][0] >= 2 else 'NOT MET'} ({count[h][0]} of 3{', STRONG' if count[h][1] >= 2 else ''}"
        f"{f', strong on {count[h][1]}' if 0 < count[h][1] < 2 else ''})" for h in HYP)
    res["headline"] = head
    res["multiplicity"] = "H1, H2, H3a and H3b are separate claims with no correction; meeting one says nothing about another."
    res["interval_limit"] = ("95% intervals: paired bootstrap over test examples (10,000 resamples, default_rng([20261021, "
                             "set index])); they leave out the variance between training seeds, so the per-seed differences "
                             "and seed SDs are given beside every TM number.")
    with open(outj, "x") as f:
        f.write(json.dumps(res, indent=1) + "\n")
    with open(outt, "x") as f:
        f.write(head + "\n" + res["multiplicity"] + "\n" + res["interval_limit"] + "\n" + "\n".join(lines) + "\n")
    print(head)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
