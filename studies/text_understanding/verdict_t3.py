#!/usr/bin/env python
"""REGISTRATION_T3.md on the rows: part 1 bars (results_flat3.jsonl, MASSIVE test), the phase A selection and
gate (results_tok.jsonl, val, seed 1) and, when its rows exist, part 2 bars (results_tok.jsonl, tags M2_*).
Prints only; writes nothing."""
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
FLAT_EXT_VAL = {"clinc150": 0.9460, "banking77": 0.9146, "hwu64": 0.9123}  # confirm-run val, existsT, seeds 1-3
CANDIDATES = ("G+S+P+N", "S+P+N", "S", "G+S")


def rows(name):
    p = HERE / name
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


# part 1
acc = defaultdict(dict)
for r in rows("results_flat3.jsonl"):
    acc[r["view"]][r["seed"]] = r["test_acc"]
m = {v: sum(a.values()) / len(a) for v, a in acc.items() if a}
print("PART 1, massive_en test, mean over seeds:", {v: (round(100 * m[v], 2), sorted(acc[v])) for v in m})
if all(len(acc[v]) == 3 for v in ("mean", "mean+existsT", "existsT", "minilm")):
    gap = m["minilm"] - m["mean"]
    print(f"  R1 works {m['mean+existsT'] - m['mean'] >= 0.005}  R1 strong {gap <= 0 or (m['mean+existsT'] - m['mean']) >= gap / 2}"
          f"  R2 works {m['existsT'] - m['mean'] >= -0.01}  R2 strong {m['existsT'] >= m['mean']}  (gap mean->minilm {100 * gap:+.2f})")
else:
    print("  part 1 incomplete")

# phase A selection and gate
A = {(r["dataset"], r["parts"]): r["val_acc"] for r in rows("results_tok.jsonl") if r["tag"].startswith("A_")}
print("PHASE A val gains over flat existsT (points):")
avg = {}
for p in CANDIDATES + ("G", "W"):
    g = {d: 100 * (A[(d, p)] - FLAT_EXT_VAL[d]) for d in FLAT_EXT_VAL if (d, p) in A}
    print(f"  {p:8s} " + "  ".join(f"{d} {v:+.2f}" for d, v in g.items()))
    if p in CANDIDATES and len(g) == 3:
        avg[p] = (sum(g.values()) / 3, sum(v >= 0.3 for v in g.values()))
if len(avg) == len(CANDIDATES):
    best = max(avg, key=lambda p: avg[p][0])
    print(f"  P* = {best} (average gain {avg[best][0]:+.2f}, >= +0.3 on {avg[best][1]} of 3): gate {'PASSED' if avg[best][1] >= 2 else 'NOT PASSED'}")
else:
    print(f"  phase A incomplete: {sorted(avg)} of {CANDIDATES} have all three datasets")

# part 2 (M2_GSPN_*, G control M3_G_* at 40 epochs; W dropped by the amendment) and REGISTRATION_T3b.md as amended
# (M3c_G_*, M3c_V_*: trained to convergence), MASSIVE test
def test_rows(prefix):
    T = defaultdict(dict)
    for r in rows("results_tok.jsonl"):
        if r["tag"].startswith(prefix) and "test_acc" in r:
            T[r["parts"]][r["seed"]] = r["test_acc"]
    return T, {p: sum(a.values()) / len(a) for p, a in T.items() if a}


T2, t2 = test_rows(("M2_", "M3_"))
print("PART 2 rows (test):", {p: (round(100 * v, 2), sorted(T2[p])) for p, v in t2.items()})
if len(T2["G+S+P+N"]) == 3 and len(T2["G"]) == 3 and "existsT" in m:
    P = t2["G+S+P+N"]
    print(f"  H3 works {P >= m['existsT'] + 0.005} (P* {100 * P:.2f} vs flat existsT {100 * m['existsT']:.2f} + 0.5)"
          f"  H3 strong {P >= m['mean+existsT'] + 0.005} (vs flat mean+existsT {100 * m['mean+existsT']:.2f} + 0.5)"
          f"  H4 not tested (amendment)  control |G - flat existsT| = {100 * abs(t2['G'] - m['existsT']):.2f}")
else:
    print("  part 2 incomplete")
T3, t3 = test_rows("M3c_")
print("T3b rows (test, to convergence):", {p: (round(100 * v, 2), sorted(T3[p])) for p, v in t3.items()},
      "stopped epochs:", {r["tag"]: r.get("stopped_epoch") for r in rows("results_tok.jsonl") if r["tag"].startswith("M3c_")})
if len(T3["G"]) == 3 and len(T3["V"]) == 3:
    a = t3["G"] - t3["V"]
    print(f"  H4a works {a >= 0.02} strong {a >= 0.05} (G - V {100 * a:+.2f})  H4b dropped (amendment)")
else:
    print("  T3b incomplete")
