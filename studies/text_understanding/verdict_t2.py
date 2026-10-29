#!/usr/bin/env python
"""Bars of REGISTRATION_T2.md on results_confirm.jsonl: test accuracy, mean of seeds 1 to 3 per (dataset, view).
Prints the table and the four verdicts; a dataset counts only when all 3 seeds of all 4 views are in.
Writes nothing."""
import json
from collections import defaultdict
from pathlib import Path

VIEWS = ("mean", "mean+existsT", "existsT", "minilm")
DATASETS = ("clinc150", "banking77", "hwu64")
acc = defaultdict(dict)
for line in Path(__file__).with_name("results_confirm.jsonl").read_text().splitlines():
    r = json.loads(line)
    acc[(r["dataset"], r["view"])][r["seed"]] = r["test_acc"]

done, m = [], {}
print(f"{'dataset':10s} " + " ".join(f"{v:>13s}" for v in VIEWS) + "   seeds")
for d in DATASETS:
    seeds = [len(acc[(d, v)]) for v in VIEWS]
    for v in VIEWS:
        a = acc[(d, v)]
        m[(d, v)] = sum(a.values()) / len(a) if a else float("nan")
    print(f"{d:10s} " + " ".join(f"{100 * m[(d, v)]:13.2f}" for v in VIEWS) + f"   {seeds}")
    if all(s == 3 for s in seeds):
        done.append(d)

print("complete datasets:", done)
rows = []
for d in done:
    base, mex, ex, mini = (m[(d, v)] for v in VIEWS)
    gap = mini - base
    rows.append({"dataset": d, "mexT_minus_mean": 100 * (mex - base), "exT_minus_mean": 100 * (ex - base),
                 "gap_closed": (mex - base) / gap if gap > 0 else float("nan")})
for r in rows:
    print(f"{r['dataset']:10s} mean+existsT {r['mexT_minus_mean']:+.2f}  existsT {r['exT_minus_mean']:+.2f}  "
          f"share of the mean->minilm gap closed by mean+existsT {r['gap_closed']:.2f}")
n = len(rows)
h1 = sum(r["mexT_minus_mean"] >= 0.5 for r in rows) >= 2 and all(r["mexT_minus_mean"] >= -0.5 for r in rows)
h1s = sum(r["gap_closed"] >= 0.5 for r in rows) >= 2
h2 = sum(r["exT_minus_mean"] >= -1.0 for r in rows) >= 2
h2s = sum(r["exT_minus_mean"] >= 0.0 for r in rows) >= 2
print(f"on {n} complete datasets: H1 works {h1}, H1 strong {h1s}, H2 works {h2}, H2 strong {h2s}")
