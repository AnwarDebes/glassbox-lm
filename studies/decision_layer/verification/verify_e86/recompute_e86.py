"""Independent recompute of e86 (decision_layer dev plan logs/plan_e86_clause_drop.md, dated note
logs/plan_e86_note_2026-10-25.md), written from the plan text; it imports nothing from the study.

    OMP_NUM_THREADS=1 nice -n 19 <python with numpy> -I recompute_e86.py <decision_layer study dir> --out <json>

- Rows: results/explore/abcd.jsonl, used only for tag, seed, time, cfg and the acc/history fields that are compared.
  For each cell and seed exactly one row must be eligible: e84 rows written after e84's freeze, e86 rows after e86's
  freeze, and for STAR (the note) after the note's freeze.
- Predictions: results/explore/<e84_preds|e86_preds>/<tag>_seed<k>.npz, preds [21, n] (20 evaluations, then the final
  call on the best-dev snapshot) and y_val. They must give the dev labels of data/<ds>/examples_dev.json.gz.
- Per cell, seeds 1 and 2: best-dev = accuracy of the last prediction; last-10 = mean accuracy of evaluations 11-20.
  Both computed from the predictions, then checked against the row (acc, history).
- Rules (points, means of seeds 1-2): keep drop 0.5 on a dataset if (drop cell - reference) >= +0.5 at best-dev and on
  the last-10 mean; E86.1 = kept on all three; E86.3 = (random + drop 0.5) - best setting >= -0.5 on both readings on
  all three, the best setting being the drop cell where kept, else the reference; on MultiWOZ the one-setting cell is
  the drop cell. TM minus LR per cell against the LR dev rows (no rule).
- Beside, no rule: a dialogue-level paired bootstrap (PCG64 seed 20261025, 10,000 resamples) of each rule difference,
  on the seed-mean per-turn correctness.
"""
import argparse
import gzip
import json
import os

import numpy as np

CELLS = {
    "multiwoz": {"ref": ("e84_multiwoz_C4_c16000_s8_nrand", "e84_preds", "e84"),
                 "drop": ("e86_multiwoz_C4_c16000_s8_nrand_drop50", "e86_preds", "e86")},
    "star": {"ref": ("e86_star_C4_c16000_s16_ref", "e86_preds", "note"),
             "drop": ("e86_star_C4_c16000_s16_drop50", "e86_preds", "note"),
             "one": ("e86_star_C4_c16000_s16_nrand_drop50", "e86_preds", "note")},
    "abcd": {"ref": ("e86_abcd_C4_c4000_s16_ref", "e86_preds", "e86"),
             "drop": ("e86_abcd_C4_c4000_s16_drop50", "e86_preds", "e86"),
             "one": ("e86_abcd_C4_c4000_s16_nrand_drop50", "e86_preds", "e86")},
}
LR = {"multiwoz": "dry_e77mwoz_lr_mem_c4", "star": "dry_R21star_lr_mem_c4", "abcd": "dry_R21abcd_lr_mem_c4"}
FREEZES = {"e84": "logs/plan_e84_negatives_fair.frozen.json", "e86": "logs/plan_e86_clause_drop.frozen.json",
           "note": "logs/plan_e86_note_2026-10-25.frozen.json"}
SEEDS = (1, 2)
B = 10_000


def dev(root, ds):
    with gzip.open(os.path.join(root, "data", ds, "examples_dev.json.gz"), "rt") as f:
        rows = json.load(f)["rows"]
    return np.array([r["y"] for r in rows]), np.array([str(r["convo_id"]) for r in rows])


def rows_by(root):
    out = {}
    with open(os.path.join(root, "results", "explore", "abcd.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            if d.get("arm") == "tm" and d.get("tag"):
                out.setdefault((d["tag"], d["seed"]), []).append(d)
    return out


def boot(diff, groups, rng):
    ug, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=diff, minlength=len(ug))
    cnts = np.bincount(inv, minlength=len(ug)).astype(float)
    idx = rng.integers(0, len(ug), size=(B, len(ug)))
    st = sums[idx].sum(1) / cnts[idx].sum(1)
    return [float(x) for x in np.percentile(st, [2.5, 97.5])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    fz = {k: json.load(open(os.path.join(root, p)))["frozen_utc"] for k, p in FREEZES.items()}
    rows = rows_by(root)
    rng = np.random.Generator(np.random.PCG64(20261025))
    res, problems, cells = {"freezes": fz}, [], {}
    for ds, cs in CELLS.items():
        y, conv = dev(root, ds)
        n = len(y)
        lr = json.load(open(os.path.join(root, "results", "explore", "dryrun", LR[ds], "seed0.json")))["dev_acc"]
        for cell, (tag, folder, rec) in cs.items():
            per, corr_best, corr_l10 = [], [], []
            for sd in SEEDS:
                cand = [r for r in rows.get((tag, sd), []) if r["time"] > fz[rec]]
                if len(cand) != 1:
                    problems.append(f"{tag} seed {sd}: {len(cand)} eligible rows (after {rec} {fz[rec]})")
                    continue
                r = cand[0]
                z = np.load(os.path.join(root, "results", "explore", folder, f"{tag}_seed{sd}.npz"))
                P, yv = z["preds"].astype(np.int64), z["y_val"].astype(np.int64)
                if not np.array_equal(yv, y) or P.shape != (21, n):
                    problems.append(f"{tag} seed {sd}: preds shape {P.shape} or labels differ from dev")
                    continue
                accs = (P == y[None, :]).mean(1)
                hist = [e["val_acc"] for e in r["history"]]
                if len(hist) != 20 or np.max(np.abs(accs[:20] - np.array(hist))) > 1e-6 or abs(accs[20] - r["acc"]) > 1e-12:
                    problems.append(f"{tag} seed {sd}: predictions do not give the row's history or acc")
                per.append({"seed": sd, "best_dev": float(accs[20]), "final": float(accs[19]), "last10": float(accs[10:20].mean()),
                            "row_time": r["time"], "neg_sampling": r["cfg"].get("neg_sampling"), "clause_drop": r["cfg"].get("clause_drop")})
                corr_best.append((P[20] == y).astype(float))
                corr_l10.append((P[10:20] == y[None, :]).mean(0))
            if len(per) == len(SEEDS):
                cells[(ds, cell)] = {"per_seed": per, "best_dev": float(np.mean([p["best_dev"] for p in per])),
                                     "last10": float(np.mean([p["last10"] for p in per])),
                                     "tm_minus_lr_points": float(100 * (np.mean([p["best_dev"] for p in per]) - lr)),
                                     "_cb": np.mean(corr_best, 0), "_cl": np.mean(corr_l10, 0)}
        res[f"{ds}_lr_dev"] = lr
        res[f"{ds}_conv"] = conv
    if problems:
        print(json.dumps({"problems": problems}, indent=1))
    kept, e863 = [], []
    out_cells = {}
    for ds, cs in CELLS.items():
        conv = res.pop(f"{ds}_conv")
        ref, drop = cells.get((ds, "ref")), cells.get((ds, "drop"))
        if ref is None or drop is None:
            continue
        b = 100 * (drop["best_dev"] - ref["best_dev"])
        l10 = 100 * (drop["last10"] - ref["last10"])
        k = b >= 0.5 - 1e-9 and l10 >= 0.5 - 1e-9
        kept += [ds] if k else []
        res[f"{ds}_drop_minus_ref"] = {"best_dev_points": b, "last10_points": l10, "kept": k,
                                       "boot_best_dev_points": [100 * x for x in boot(drop["_cb"] - ref["_cb"], conv, rng)],
                                       "boot_last10_points": [100 * x for x in boot(drop["_cl"] - ref["_cl"], conv, rng)]}
        best = drop if k else ref
        one = cells.get((ds, "one"), drop)
        b3 = 100 * (one["best_dev"] - best["best_dev"])
        l3 = 100 * (one["last10"] - best["last10"])
        h = b3 >= -0.5 - 1e-9 and l3 >= -0.5 - 1e-9
        e863.append(h)
        res[f"{ds}_one_minus_best"] = {"best_setting": "drop" if k else "ref", "best_dev_points": b3, "last10_points": l3, "holds": h,
                                       "boot_best_dev_points": [100 * x for x in boot(one["_cb"] - best["_cb"], conv, rng)] if one is not best else [0.0, 0.0]}
        for cell in cs:
            c = cells[(ds, cell)]
            out_cells[f"{ds}/{cell}"] = {k2: v for k2, v in c.items() if not k2.startswith("_")}
    res["cells"] = out_cells
    res["kept_on"] = kept
    res["E86.1"] = "MET" if len(kept) == 3 else "not met"
    res["E86.3"] = "MET" if (len(e863) == 3 and all(e863)) else "not met"
    res["problems"] = problems
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "cells"}, indent=1))


if __name__ == "__main__":
    main()
