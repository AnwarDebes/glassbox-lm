"""Independent recompute of REGISTRATION_N7.md's bars from results_n7_test.jsonl (read only).

Written from the registration text, not from verdict_n7.py:
- S1 = per run the mean test accuracy over epochs 51-100 (field test_mean_last50); S2 = test accuracy at the best
  carved-val epoch (field test_at_best_val). Means over seeds 1-5; a view counts only with all 5 seeds; the first of
  repeated (dataset, view, seed) rows is kept.
- A: |bow S1 - Yadav BoW| <= 2.0 on >= 3 of TREC, MR, R52, R8 (two-sided).
- B: bow+existsT_glove S1 > Yadav's best TM+GloVe row on >= 3 of 4.
- C: bow+existsT_potion S1 > the same rows on >= 3 of 4; strong: also > the best stated-protocol TM number on >= 3 of 4.
- D_potion / D_glove: on S2, paired by seed, bow+existsT_X - bow "adds" on a set when the mean > 0 and >= 4 of 5 seed
  differences > 0; met when it adds on >= 3 of the 5 sets. E: bow+existsT_potion - bow+existsT_glove, same rule.
Also recomputes S1 and S2 from the stored curves when present, as a check of the stored fields.
"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))
import json
import sys
from pathlib import Path

import statistics as st

RES = Path(sys.argv[1] if len(sys.argv) > 1 else f"{_REPO_ROOT}/studies/text_understanding/results_n7_test.jsonl")
SEEDS = (1, 2, 3, 4, 5)
SETS = ("trec", "mr", "sst2", "r52", "r8")
BARS4 = ("trec", "mr", "r52", "r8")
YADAV_BOW = {"trec": 88.05, "mr": 75.14, "r52": 84.62, "r8": 96.16}
YADAV_GLOVE = {"trec": 90.04, "mr": 77.51, "r52": 89.14, "r8": 97.50}
BEST_TM = {"trec": 91.61, "mr": 77.51, "r52": 89.53, "r8": 97.50}
EPS = 1e-9

rows = {}
for line in RES.read_text().splitlines():
    if not line.strip():
        continue
    r = json.loads(line)
    if r.get("registration") != "REGISTRATION_N7.md" or r.get("mode") != "test":
        continue
    k = (r["dataset"], r["view"], int(r["seed"]))
    rows.setdefault(k, r)
print(f"{len(rows)} unique test rows")

# check the stored statistics against the curves, if the curves are stored
bad = 0
for k, r in rows.items():
    c = r.get("curves") or {}
    te, va = c.get("test"), c.get("val")
    if isinstance(te, list) and len(te) >= 100:
        s1 = 100 * sum(te[50:100]) / 50 if max(te) <= 1.0 else sum(te[50:100]) / 50
        if abs(s1 - r["test_mean_last50"]) > 0.01:
            bad += 1
    if isinstance(te, list) and isinstance(va, list) and len(va) == len(te):
        best = max(range(len(va)), key=lambda i: (va[i], -i))
        s2 = te[best] * (100 if max(te) <= 1.0 else 1)
        if abs(s2 - r["test_at_best_val"]) > 0.01:
            bad += 1
print(f"stored S1/S2 that disagree with the curves: {bad}")


def mean(ds, v, key):
    a = [rows[(ds, v, s)][key] for s in SEEDS if (ds, v, s) in rows]
    return st.mean(a) if len(a) == 5 else None


for ds in SETS:
    print(f"\n{ds}")
    for v in ("bow", "existsT_potion", "existsT_glove", "bow+existsT_potion", "bow+existsT_glove"):
        s1, s2 = mean(ds, v, "test_mean_last50"), mean(ds, v, "test_at_best_val")
        print(f"  {v:20s} S1 {s1 if s1 is None else round(s1, 2)}  S2 {s2 if s2 is None else round(s2, 2)}")

a = b = c = cs = 0
for ds in BARS4:
    bow, gl, po = (mean(ds, v, "test_mean_last50") for v in ("bow", "bow+existsT_glove", "bow+existsT_potion"))
    if bow is not None:
        a += abs(bow - YADAV_BOW[ds]) <= 2.0 + EPS
    if gl is not None:
        b += gl > YADAV_GLOVE[ds] + EPS
    if po is not None:
        c += po > YADAV_GLOVE[ds] + EPS
        cs += po > BEST_TM[ds] + EPS
print(f"\nA {a}/4 -> {'HOLDS' if a >= 3 else 'DOES NOT HOLD'}; B {b}/4 -> {'MET' if b >= 3 else 'NOT MET'}; "
      f"C {c}/4 -> {'MET' if c >= 3 else 'NOT MET'}, strong {cs}/4 -> {'MET' if c >= 3 and cs >= 3 else 'NOT MET'}")
for name, va, vb in (("D_potion", "bow+existsT_potion", "bow"), ("D_glove", "bow+existsT_glove", "bow"),
                     ("E", "bow+existsT_potion", "bow+existsT_glove")):
    n = 0
    det = []
    for ds in SETS:
        d = [rows[(ds, va, s)]["test_at_best_val"] - rows[(ds, vb, s)]["test_at_best_val"]
             for s in SEEDS if (ds, va, s) in rows and (ds, vb, s) in rows]
        if len(d) < 5:
            det.append(f"{ds}: incomplete")
            continue
        adds = st.mean(d) > EPS and sum(x > EPS for x in d) >= 4
        n += adds
        det.append(f"{ds} {st.mean(d):+.2f} ({sum(x > EPS for x in d)}/5){' adds' if adds else ''}")
    print(f"{name}: adds on {n} of 5 -> {'MET' if n >= 3 else 'NOT MET'}  [{'; '.join(det)}]")
