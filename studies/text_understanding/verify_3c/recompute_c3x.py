#!/usr/bin/env python
"""Independent recompute of the c3x report (tu/c3x/report_c3x.json), written from the frozen note's definitions
(tu/c3x/NOTE_2026-10-21_followups_v2.md), not from c3x_report.py. CPU only, reads files only.

    python -I -B verify_3c/recompute_c3x.py

From the raw rows (tu/c3x/results_c3x_val.jsonl tag "3cx", results_c3x_lr.jsonl) and the feature metas, per set:
- TM: each arm's val accuracy (points) for seeds 4-6; gains = the mean of the per-seed differences, with paired
  SE = sd(ddof 1) / sqrt 3; rep5 = orig2 with repeat 5;
- LR: the best-C val accuracy, the per-C accuracies, and the gains at C 1 and C 10;
- one k-means: every arm's meta has the same kmeans_run_id;
- replication: clearly smaller when the TM gain F2 - orig2 is below the registered gain (tu/c3/verdict_c3.json) minus 2
  of its SEs, or the LR gain is at least 0.3 lower.
Every recomputed number is compared with the report (tolerance 1e-6). Writes verify_3c/recompute_c3x_<UTC>.json.
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))

import json
import math
import sys
import time
from pathlib import Path

TU = Path(f"{_REPO_ROOT}/studies/text_understanding/tu")
CX = TU / "c3x"
SETS = ("clinc150", "banking77", "hwu64")
SEEDS = (4, 5, 6)


def paired(a: list[float], b: list[float]) -> tuple[float, float, list[float]]:
    d = [x - y for x, y in zip(a, b)]
    m = sum(d) / len(d)
    return m, math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1)) / math.sqrt(len(d)), d


assert all(abs(u - v) < 1e-12 for u, v in zip(paired([3, 4, 5], [1, 1, 1])[:2], [3.0, 1 / math.sqrt(3)]))  # self-check


def main() -> int:
    tm = [r for r in map(json.loads, open(CX / "results_c3x_val.jsonl")) if r.get("tag") == "3cx"]
    lr = list(map(json.loads, open(CX / "results_c3x_lr.jsonl")))
    rep = json.load(open(CX / "report_c3x.json"))
    reg = json.load(open(TU / "c3" / "verdict_c3.json"))
    checks, bad, out = 0, [], {}

    def cmp(name: str, mine: float | bool | None, theirs) -> None:
        nonlocal checks
        checks += 1
        same = (mine == theirs) if isinstance(mine, bool) or mine is None or theirs is None else abs(mine - theirs) < 1e-6
        if not same:
            bad.append(f"{name}: recomputed {mine}, report {theirs}")

    for d in SETS:
        R = rep["sets"][d]
        arms = ["orig2", "F2", "twin"] + (["F_rk"] if d == "clinc150" else [])
        acc = {}
        for arm in arms + ["rep5"]:
            a, k = ("orig2", 5) if arm == "rep5" else (arm, 1)
            rows = {r["seed"]: r for r in tm if r["dataset"] == d and r["arm"] == a and r.get("repeat", 1) == k}
            assert sorted(rows) == list(SEEDS), (d, arm, sorted(rows))
            acc[arm] = [round(100 * rows[s]["val_acc"], 4) for s in SEEDS]
            for i, s in enumerate(SEEDS):
                cmp(f"{d} tm {arm} seed {s}", acc[arm][i], R["tm"][arm][i])
        runs = {json.load(open(CX / "feats" / arm / f"{d}__p32tok_K512.meta.json"))["kmeans_run_id"] for arm in arms}
        cmp(f"{d} one k-means", len(runs) == 1, R["one_kmeans"])
        pairs = [("F2", "orig2"), ("twin", "orig2"), ("F2", "twin"), ("rep5", "orig2"), ("F2", "rep5")]
        if d == "clinc150":
            pairs += [("F_rk", "orig2"), ("F2", "F_rk")]
        g = {}
        for x, y in pairs:
            m, se, per = paired(acc[x], acc[y])
            g[f"{x}-{y}"] = (m, se)
            cmp(f"{d} tm {x}-{y} mean", m, R["tm_gains"][f"{x}-{y}"]["mean"])
            cmp(f"{d} tm {x}-{y} se", se, R["tm_gains"][f"{x}-{y}"]["se"])
        L = {r["name"]: r for r in lr if r["dataset"] == d and r["name"] in arms}
        assert sorted(L) == sorted(arms), (d, sorted(L))
        for arm in arms:
            best = max(L[arm]["acc_by_C"].items(), key=lambda kv: kv[1])  # the first strictly highest, in grid order
            first = next(c for c, v in L[arm]["acc_by_C"].items() if v == best[1])
            cmp(f"{d} lr {arm} best = lr_val", L[arm]["acc_by_C"][first], L[arm]["lr_val"])
            cmp(f"{d} lr {arm}", round(100 * L[arm]["lr_val"], 4), R["lr"][arm])
        lrg = {f"{x}-{y}": 100 * (L[x]["lr_val"] - L[y]["lr_val"]) for x, y in pairs if x in L and y in L}
        for k2, v in lrg.items():
            cmp(f"{d} lr gain {k2}", v, R["lr_gains"][k2])
        for c in ("1.0", "10.0"):
            for x, y in pairs:
                if x in L and y in L:
                    cmp(f"{d} lr C {c} {x}-{y}", 100 * (L[x]["acc_by_C"][c] - L[y]["acc_by_C"][c]), R["lr_gains_fixed_C"][c][f"{x}-{y}"])
        rt, rl = reg[d]["tm_gain"], reg[d]["lr_gain"]
        m, se = g["F2-orig2"]
        smaller = (m < rt - 2 * se) or (lrg["F2-orig2"] <= rl - 0.3 + 1e-9)
        cmp(f"{d} replication clearly smaller", smaller, R["replication"]["clearly_smaller"])
        out[d] = {"tm_gains": {k2: [round(v[0], 4), round(v[1], 4)] for k2, v in g.items()},
                  "lr_gains": {k2: round(v, 4) for k2, v in lrg.items()}, "clearly_smaller": smaller}
    res = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "checks": checks, "mismatches": bad,
           "verdict": "MATCH" if not bad else "MISMATCH", "sets": out}
    p = Path(__file__).resolve().parent / f"recompute_c3x_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    with open(p, "x") as f:
        f.write(json.dumps(res, indent=1) + "\n")
    print(f"{res['verdict']}: {checks} checks, {len(bad)} mismatches -> {p}")
    for b in bad:
        print("  " + b)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
