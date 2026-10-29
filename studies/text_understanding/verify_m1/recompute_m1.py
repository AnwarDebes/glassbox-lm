#!/usr/bin/env python
"""Independent recompute of M1's verdict (tu/PLAN_M1.md, registered), written from the plan's text, not from
m1_verdict.py. CPU only; reads the saved test predictions, the real test labels and the rows.

    python -I -B verify_m1/recompute_m1.py

Per set: each configuration's per-seed test accuracy from its predictions (seeds 7-9; LR one fit), the hypotheses on the
point estimates (H1 a3B-C4 >= +0.5 TM and LR; H2 a3B-potion >= -0.5 TM and LR; H3a LR a3B_3c-a3B >= +0.5; H3b LR
a3B_3c-a3B_twin >= +0.5), and the 95% intervals by the registered scheme (paired bootstrap over test examples, 10,000
resamples, numpy default_rng([20261021, set index]), one index draw per resample shared by all comparisons and seeds;
drawn here in blocks of 1,000). Everything is compared with tu/m1/verdict_m1.json. Writes
verify_m1/recompute_m1_<UTC>.json.
"""
from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, f"{_REPO_ROOT}/studies/decision_layer/src")
from dl.data import load  # noqa: E402

M1 = Path(f"{_REPO_ROOT}/studies/text_understanding/tu/m1")
SETS, SEEDS = ("clinc150", "banking77", "hwu64"), (7, 8, 9)
TMC = [("C4", 1), ("a3B", 1), ("potion", 1), ("a3B_3c", 1), ("C4_3c", 1), ("a3B_twin", 1), ("a3B", 5)]
LRA = ["C4", "a3B", "potion", "a3B_3c", "C4_3c", "a3B_twin"]
HYP = {"H1": [("tm", ("a3B", 1), ("C4", 1), 0.5), ("lr", "a3B", "C4", 0.5)],
       "H2": [("tm", ("a3B", 1), ("potion", 1), -0.5), ("lr", "a3B", "potion", -0.5)],
       "H3a": [("lr", "a3B_3c", "a3B", 0.5)], "H3b": [("lr", "a3B_3c", "a3B_twin", 0.5)]}


def main() -> int:
    tm = [r for r in map(json.loads, open(M1 / "results_m1_tm.jsonl")) if r.get("tag") == "m1"]
    lr = list(map(json.loads, open(M1 / "results_m1_lr.jsonl")))
    V = json.load(open(M1 / "verdict_m1.json"))
    bad, n_checks, out = [], 0, {}
    for si, d in enumerate(SETS):
        y = load(d).labels["test"]
        C = {}
        for arm, rep in TMC:
            rows = {r["seed"]: r for r in tm if r["dataset"] == d and r["arm"] == arm and r["repeat"] == rep}
            assert sorted(rows) == list(SEEDS), (d, arm, rep)
            C[("tm", arm, rep)] = np.stack([np.load(rows[s]["preds"]) == y for s in SEEDS]).astype(float)
        for arm in LRA:
            (r,) = [r for r in lr if r["dataset"] == d and r["name"] == arm]
            C[("lr", arm)] = (np.load(r["preds"]) == y).astype(float)
        diffs = {}
        for h, parts in HYP.items():
            for probe, a, b, bar in parts:
                if probe == "tm":
                    diffs[(h, probe)] = (C[("tm", *a)] - C[("tm", *b)]).mean(0), bar
                else:
                    diffs[(h, probe)] = C[("lr", a)] - C[("lr", b)], bar
        rng = np.random.default_rng([20261021, si])
        n = len(y)
        boot = {k: [] for k in diffs}
        for _ in range(10):
            idx = rng.integers(0, n, size=(1000, n))
            for k, (v, _bar) in diffs.items():
                boot[k].append(v[idx].mean(1))
        res = {}
        for h, parts in HYP.items():
            pts = {p[0]: 100 * diffs[(h, p[0])][0].mean() for p in parts}
            los = {p[0]: float(np.percentile(100 * np.concatenate(boot[(h, p[0])]), 2.5)) for p in parts}
            met = all(pts[p[0]] >= p[3] - 1e-9 for p in parts)
            strong = all(los[p[0]] >= p[3] - 1e-9 for p in parts)
            res[h] = {"met": met, "strong": strong, "points": {k: round(v, 4) for k, v in pts.items()},
                      "lower": {k: round(v, 4) for k, v in los.items()}}
            vh = V["sets"][d]["hypotheses"][h]
            n_checks += 2
            if vh["met"] != met or vh["strong"] != strong:
                bad.append(f"{d} {h}: recomputed met {met} strong {strong}; verdict {vh}")
            for p in parts:
                name = f"{h}_{p[0]}"
                vc = V["sets"][d]["comparisons"][name]
                n_checks += 2
                if abs(vc["point"] - round(pts[p[0]], 4)) > 1e-3:
                    bad.append(f"{d} {name} point {pts[p[0]]:.4f} vs {vc['point']}")
                if abs(vc["ci95"][0] - los[p[0]]) > 1e-3:
                    bad.append(f"{d} {name} lower {los[p[0]]:.4f} vs {vc['ci95'][0]}")
        out[d] = res
    cnt = {h: sum(out[d][h]["met"] for d in SETS) for h in HYP}
    head = {h: {"met": cnt[h] >= 2, "sets_met": cnt[h]} for h in HYP}
    for h in HYP:
        n_checks += 1
        if V["hypotheses"][h]["sets_met"] != cnt[h]:
            bad.append(f"{h} sets met {cnt[h]} vs {V['hypotheses'][h]['sets_met']}")
    res = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "checks": n_checks, "mismatches": bad,
           "verdict": "MATCH" if not bad else "MISMATCH", "hypotheses": head, "sets": out}
    p = Path(__file__).resolve().parent / f"recompute_m1_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    with open(p, "x") as f:
        f.write(json.dumps(res, indent=1) + "\n")
    print(f"{res['verdict']}: {n_checks} checks, {len(bad)} mismatches -> {p}")
    for b in bad:
        print("  " + b)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
