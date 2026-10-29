#!/usr/bin/env python
"""Step 3d, the verdict (tu/PLAN_3d.md "Bar" and "Attribution"): reads the 3d rows and prints the registered verdict,
the attribution rules and the descriptive lines. VAL only.

    python -I -B d3_verdict.py --d3 tu/d3 [--m1 tu/m1]   # writes <d3>/verdict_d3.json and verdict_d3.txt

Registered (the only gain claim): J - a3B >= +0.5 points with BOTH probes (TM: the mean of seeds 7-9 at s 8; LR: val at
the C picked on val) on at least 2 of the 3 sets -> PASS, else FAIL. Gains come from the counts of correct val
predictions in the rows (val_correct, lr_correct over n_val), so they are exact; met = gain >= 0.5 - 1e-9.
Attribution rules (fixed with the bar; each needs +0.5 with both probes on at least 2 of 3 sets; they only license an
attribution sentence, never a second gain claim; no multiplicity correction, each stands alone):
  A1 BJ - MID   "the fitted pair rows carry meaning beyond their words' midpoints" (a3B's word rows in both);
  A2 Jwc - JnoP "the pair tokens add, at J's word codebook";
  A3 JnoP - a3B "the joint refit of the word rows adds".
Descriptive only: every other difference of the map (J - Jwc the codebook, MID - a3B extra tokens with no new meaning,
BJ - MW fitted against embedded alone, J - BJ, J - MW); J - a3B at the mean of the last three evaluations; a two-level
bootstrap interval for J - a3B (TM: seeds within each arm, then val items, shared; LR: val items; 10,000 resamples,
numpy default_rng([20261023, set index])); the k-means spread (a3B and J at k-means seeds 0, 1, 2, TM seed 7, and LR);
per-arm per-seed values (the same seed number on different features is not a pairing); the a3B reference beside M1's
a3B val rows (TM and LR; a reproduction check, no rule); the pair coverage.
Refuses (exit 2, nothing written) when a row is missing, doubled, a smoke row, of another study, view, cfg, seed set or
table, lacks its counts, or when an arm's rows do not share one k-means run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SETS = ("clinc150", "banking77", "hwu64")
MAIN = ("a3B", "J", "Jwc", "JnoP", "MW", "MID", "BJ")
KARMS = ("a3B_k1", "a3B_k2", "J_k1", "J_k2")
SEEDS = (7, 8, 9)
KSEED = 7
BAR = 0.5
CFG = {"n_clauses": 4000, "T": 4000, "s": 8.0, "epochs": 40, "batch_size": 128}
TABLE = {"a3B": "a3B_d768", "J": "d3J_d768", "Jwc": "d3J_d768", "JnoP": "d3J_d768", "MW": "d3MW_d768",
         "MID": "d3MID_d768", "BJ": "d3BJ_d768", "a3B_k1": "a3B_d768", "a3B_k2": "a3B_d768", "J_k1": "d3J_d768",
         "J_k2": "d3J_d768"}
RULES = {"A1": ("BJ", "MID"), "A2": ("Jwc", "JnoP"), "A3": ("JnoP", "a3B")}
DESC = (("J", "Jwc"), ("MID", "a3B"), ("BJ", "MW"), ("J", "BJ"), ("J", "MW"), ("Jwc", "JnoP"), ("BJ", "a3B"),
        ("MW", "a3B"), ("J", "JnoP"))
B = 10_000


def refuse(msg: str) -> None:
    print(f"REFUSED: {msg}")
    sys.exit(2)


def rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def collect(d3: Path):
    tm, lr = {}, {}
    for r in rows(d3 / "results_d3_val.jsonl"):
        if r.get("tag") != "d3":
            continue
        k = (r["dataset"], r["arm"], r["seed"])
        if k in tm:
            refuse(f"doubled TM row {k}")
        if r.get("study") != "3d" or r.get("smoke") or r.get("view") != "existsT" or r.get("repeat", 1) != 1:
            refuse(f"TM row {k}: study, smoke, view or repeat")
        if any(r["cfg"].get(a) != b for a, b in CFG.items()):
            refuse(f"TM row {k}: cfg {({a: r['cfg'].get(a) for a in CFG})}")
        if not isinstance(r.get("val_correct"), int) or not r.get("n_val"):
            refuse(f"TM row {k}: no counts")
        tm[k] = r
    for r in rows(d3 / "results_d3_lr.jsonl"):
        k = (r["dataset"], r["name"])
        if k in lr:
            refuse(f"doubled LR row {k}")
        if r.get("study") != "3d" or r.get("view") != "existsT":
            refuse(f"LR row {k}: study or view")
        if not isinstance(r.get("lr_correct"), int) or not r.get("n_val"):
            refuse(f"LR row {k}: no counts")
        lr[k] = r
    for d in SETS:
        for a in MAIN + KARMS:
            seeds = SEEDS if a in MAIN else (KSEED,)
            for s in seeds:
                if (d, a, s) not in tm:
                    refuse(f"missing TM row {(d, a, s)}")
            if (d, a) not in lr:
                refuse(f"missing LR row {(d, a)}")
            group = [tm[(d, a, s)] for s in seeds] + [lr[(d, a)]]
            if any(g["arm"] != a for g in group):
                refuse(f"arm label mismatch in {(d, a)}")
            if any(g.get("table") != TABLE[a] for g in group):
                refuse(f"table mismatch in {(d, a)}: {[g.get('table') for g in group]}")
            if len({g.get("kmeans_run_id") for g in group}) != 1 or len({g["n_val"] for g in group}) != 1:
                refuse(f"{(d, a)}: rows of one arm with different k-means runs or val sizes")
    extra = {k for k in tm if k[0] in SETS and k[1] in MAIN + KARMS and k[2] not in (SEEDS if k[1] in MAIN else (KSEED,))}
    if extra:
        refuse(f"TM rows with other seeds: {sorted(extra)}")
    return tm, lr


def acc_tm(tm, d, a, seeds=SEEDS) -> float:
    return float(np.mean([tm[(d, a, s)]["val_correct"] / tm[(d, a, s)]["n_val"] for s in seeds]))


def acc_lr(lr, d, a) -> float:
    return lr[(d, a)]["lr_correct"] / lr[(d, a)]["n_val"]


def last3(tm, d, a) -> float:
    return float(np.mean([np.mean([x for _, x in tm[(d, a, s)]["curve"][-3:]]) for s in SEEDS]))


def bootstrap(tm, lr, d, i, y) -> dict:
    """Descriptive two-level interval for J - a3B (TM) and a paired item interval (LR)."""
    rng = np.random.default_rng([20261023, i])
    P = {a: np.stack([np.load(tm[(d, a, s)]["preds_val"]) for s in SEEDS]) == y[None, :] for a in ("J", "a3B")}
    L = {a: np.load(lr[(d, a)]["preds_val"]) == y for a in ("J", "a3B")}
    n = len(y)
    out_tm, out_lr = np.empty(B), np.empty(B)
    for b0 in range(0, B, 500):
        m = min(500, B - b0)
        it = rng.integers(0, n, (m, n))
        sj, sa = rng.integers(0, 3, (m, 3)), rng.integers(0, 3, (m, 3))
        cj = P["J"][:, it].mean(2)  # (3 seeds, m): each seed's accuracy on the resampled items
        ca = P["a3B"][:, it].mean(2)
        out_tm[b0 : b0 + m] = (np.take_along_axis(cj.T, sj, 1).mean(1) - np.take_along_axis(ca.T, sa, 1).mean(1))
        out_lr[b0 : b0 + m] = L["J"][it].mean(1) - L["a3B"][it].mean(1)
    q = lambda v: [round(100 * float(np.quantile(v, 0.025)), 3), round(100 * float(np.quantile(v, 0.975)), 3)]
    return {"tm_95": q(out_tm), "lr_95": q(out_lr), "resamples": B, "rng": [20261023, i]}


def verdict(d3: Path, tm: dict, lr: dict, m1_tm: list[dict], m1_lr: list[dict]) -> dict:
    out = {"bar": BAR, "sets": {}, "rules": {}}
    met = 0
    gain = {}
    for i, d in enumerate(SETS):
        T = {a: acc_tm(tm, d, a) for a in MAIN}
        L = {a: acc_lr(lr, d, a) for a in MAIN + KARMS}
        def g(x, y, T=T, L=L):  # binds this set's values (a later call must not see another set's)
            return {"tm": round(100 * (T[x] - T[y]), 4), "lr": round(100 * (L[x] - L[y]), 4)}

        gain[d] = g
        gj = g("J", "a3B")
        ok = gj["tm"] >= BAR - 1e-9 and gj["lr"] >= BAR - 1e-9
        met += ok
        yv = np.load(d3 / "feats" / "J" / f"{d}__p32tok_K512.npz")["y_val"]
        ks = {a: tm[(d, a, KSEED)]["val_correct"] / tm[(d, a, KSEED)]["n_val"] for a in MAIN + KARMS if (d, a, KSEED) in tm}
        e = {"J_minus_a3B": gj, "met": bool(ok),
             "tm": {a: round(100 * T[a], 3) for a in MAIN}, "lr": {a: round(100 * L[a], 3) for a in MAIN},
             "lr_C": {a: lr[(d, a)]["C"] for a in MAIN},
             "tm_per_seed": {a: [round(100 * tm[(d, a, s)]["val_correct"] / tm[(d, a, s)]["n_val"], 2) for s in SEEDS]
                             for a in MAIN},
             "descriptive": {f"{x}_minus_{y}": g(x, y) for x, y in DESC},
             "J_minus_a3B_last3_tm": round(100 * (last3(tm, d, "J") - last3(tm, d, "a3B")), 3),
             "interval_J_minus_a3B": bootstrap(tm, lr, d, i, yv),
             "kmeans_spread": {"tm_seed7": {a: round(100 * ks[a], 2) for a in ("a3B", "a3B_k1", "a3B_k2", "J", "J_k1", "J_k2")},
                               "lr": {a: round(100 * L[a], 2) for a in ("a3B", "a3B_k1", "a3B_k2", "J", "J_k1", "J_k2")},
                               "J_minus_a3B_by_kmeans_seed": {
                                   str(k): {"tm_seed7": round(100 * (ks[ja] - ks[aa]), 2), "lr": round(100 * (L[ja] - L[aa]), 2)}
                                   for k, ja, aa in ((0, "J", "a3B"), (1, "J_k1", "a3B_k1"), (2, "J_k2", "a3B_k2"))}}}
        m1t = {r["seed"]: r["val_acc"] for r in m1_tm if r.get("tag") == "m1" and r["dataset"] == d and r.get("arm") == "a3B"
               and r.get("repeat", 1) == 1}
        m1l = [r["lr_val"] for r in m1_lr if r["dataset"] == d and r.get("name") == "a3B"]
        e["a3B_vs_M1_val"] = {"tm": {str(s): [tm[(d, "a3B", s)]["val_acc"], m1t.get(s)] for s in SEEDS},
                              "lr": [lr[(d, "a3B")]["lr_val"], m1l[0] if len(m1l) == 1 else None]}
        e["a3B_equals_M1_val"] = (all(m1t.get(s) == tm[(d, "a3B", s)]["val_acc"] for s in SEEDS)
                                  and len(m1l) == 1 and m1l[0] == lr[(d, "a3B")]["lr_val"])
        out["sets"][d] = e
    out["sets_met"] = met
    out["verdict"] = "PASS" if met >= 2 else "FAIL"
    for name, (x, y) in RULES.items():
        per = {d: gain[d](x, y) for d in SETS}
        n_ok = sum(v["tm"] >= BAR - 1e-9 and v["lr"] >= BAR - 1e-9 for v in per.values())
        out["rules"][name] = {"contrast": f"{x} - {y}", "per_set": per, "sets_met": n_ok, "met": n_ok >= 2}
    return out


def text(v: dict) -> str:
    L = [(f"VERDICT 3d (val): {v['verdict']} (J - a3B >= +{v['bar']} with both probes on {v['sets_met']} of 3 sets; "
          "2 needed)")]
    for d, e in v["sets"].items():
        g = e["J_minus_a3B"]
        iv = e["interval_J_minus_a3B"]
        L.append(f"  {d}: J - a3B TM {g['tm']:+.2f} [{iv['tm_95'][0]:+.2f}, {iv['tm_95'][1]:+.2f}], LR {g['lr']:+.2f} "
                 f"[{iv['lr_95'][0]:+.2f}, {iv['lr_95'][1]:+.2f}] -> {'met' if e['met'] else 'not met'}; "
                 f"TM at the last-3 mean {e['J_minus_a3B_last3_tm']:+.2f}")
        L.append("    TM " + ", ".join(f"{a} {x:.2f}" for a, x in e["tm"].items()) + " | LR "
                 + ", ".join(f"{a} {x:.2f}" for a, x in e["lr"].items()))
        L.append("    per seed (not paired): " + "; ".join(f"{a} {x}" for a, x in e["tm_per_seed"].items() if a in ("a3B", "J")))
        L.append("    descriptive: " + "; ".join(f"{k} TM {x['tm']:+.2f} LR {x['lr']:+.2f}" for k, x in e["descriptive"].items()))
        L.append(f"    k-means spread, J - a3B by k-means seed: {e['kmeans_spread']['J_minus_a3B_by_kmeans_seed']}")
        L.append(f"    a3B reference equals M1's a3B val rows: {e['a3B_equals_M1_val']} {e['a3B_vs_M1_val']}")
    for name, r in v["rules"].items():
        L.append(f"  attribution {name} ({r['contrast']}): {'met' if r['met'] else 'not met'} ({r['sets_met']} of 3) "
                 + "; ".join(f"{d} TM {x['tm']:+.2f} LR {x['lr']:+.2f}" for d, x in r["per_set"].items()))
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d3", type=Path, required=True)
    ap.add_argument("--m1", type=Path, default=Path(__file__).resolve().parent / "m1")
    args = ap.parse_args()
    for p in (args.d3 / "verdict_d3.json", args.d3 / "verdict_d3.txt"):
        assert not p.exists(), f"{p} exists"
    tm, lr = collect(args.d3)
    v = verdict(args.d3, tm, lr, rows(args.m1 / "results_m1_tm.jsonl"), rows(args.m1 / "results_m1_lr.jsonl"))
    cov = {}
    for d in SETS:
        mp = args.d3 / "feats" / "J" / f"{d}__p32tok_K512.meta.json"
        if mp.exists():
            m = json.loads(mp.read_text())
            cov[d] = {k: m.get(k) for k in ("coverage", "train_vocab_words", "train_vocab_pairs", "concepts_with_a_pair_among_top4")}
    v["pair_coverage_J"] = cov
    (args.d3 / "verdict_d3.json").write_text(json.dumps(v, indent=1))
    t = text(v)
    (args.d3 / "verdict_d3.txt").write_text(t + "\n")
    print(t)


if __name__ == "__main__":
    main()
