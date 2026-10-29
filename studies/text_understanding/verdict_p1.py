#!/usr/bin/env python
"""Verdict of REGISTRATION_P1.md (the Model2Vec authors' low-data protocol): reads results_p1.jsonl.

    verdict_p1.py

Per dataset: the mean and sd over the draws of every row (weighted F1, points), the paired differences of the bars,
and how many draws agree in sign. A dataset counts only when all 5 draws of the rows it needs are present.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PUB = {"banking77": 79.95, "clinc_oos": 68.11, "massive_intent": 73.83}
ROWS = ("lr_mean", "lr_existsT", "tm_existsT", "tm_mean+existsT", "tm_mean")
DRAWS = (1, 2, 3, 4, 5)
res = {}
for x in ((HERE / "results_p1.jsonl").read_text().splitlines() if (HERE / "results_p1.jsonl").exists() else []):
    if x.strip():
        r = json.loads(x)
        res[(r["dataset"], r["row"], r["draw"])] = r


def vals(ds, row, key="weighted_f1"):
    v = [res[(ds, row, d)][key] for d in DRAWS if (ds, row, d) in res]
    return np.array(v) if len(v) == len(DRAWS) else None


def diff(ds, a, b):
    x, y = vals(ds, a), vals(ds, b)
    return None if x is None or y is None else x - y


bars = {k: 0 for k in ("R0", "P1a", "P1b", "P1c")}
strong_pub = True
complete = 0
for ds in PUB:
    print(f"== {ds} (published lr_mean {PUB[ds]})")
    for row in ROWS:
        v, a = vals(ds, row), vals(ds, row, "acc")
        if v is None:
            print(f"  {row:16s} incomplete")
            continue
        extra = ""
        if row.startswith("tm_"):
            extra = f", epochs {[res[(ds, row, d)]['epochs'] for d in DRAWS]}"
        print(f"  {row:16s} wF1 {v.mean():6.2f} (sd {v.std(ddof=1):.2f}; {v.round(2).tolist()}), acc {a.mean():6.2f}{extra}")
    lm = vals(ds, "lr_mean")
    if lm is None:
        continue
    complete += 1
    r0 = abs(lm.mean() - PUB[ds]) <= 2.0
    bars["R0"] += r0
    print(f"  R0 reproduction: {lm.mean():.2f} vs {PUB[ds]} ({lm.mean() - PUB[ds]:+.2f}) -> {'PASS' if r0 else 'OUTSIDE 2.0'}")
    for name, a, b in (("P1a", "lr_existsT", "lr_mean"), ("P1b", "tm_existsT", "lr_mean"), ("P1c", "tm_mean+existsT", "lr_mean"),
                       ("report", "tm_existsT", "lr_existsT"), ("report", "tm_mean", "lr_mean")):
        d = diff(ds, a, b)
        if d is None:
            print(f"  {name} {a} - {b}: incomplete")
            continue
        print(f"  {name} {a} - {b}: {d.mean():+.2f} (draws > 0: {(d > 0).sum()} of {len(d)})")
        if name in bars and d.mean() > 0:
            bars[name] += 1
    te = vals(ds, "tm_existsT")
    if te is not None and r0 and te.mean() < PUB[ds]:
        strong_pub = False
print(f"\ncomplete datasets: {complete} of 3")
print(f"R0 within 2.0 of the published number: {bars['R0']} of 3")
print(f"P1a (lr_existsT > lr_mean on >= 2 of 3): {bars['P1a']} of 3 -> {'MET' if bars['P1a'] >= 2 else 'NOT MET'}")
print(f"P1b (tm_existsT > lr_mean on >= 2 of 3): {bars['P1b']} of 3 -> {'MET' if bars['P1b'] >= 2 else 'NOT MET'}; strong "
      f"(3 of 3 and >= published where R0 passed): {'MET' if bars['P1b'] == 3 and strong_pub else 'NOT MET'}")
print(f"P1c (tm_mean+existsT > lr_mean on >= 2 of 3): {bars['P1c']} of 3 -> {'MET' if bars['P1c'] >= 2 else 'NOT MET'}")
