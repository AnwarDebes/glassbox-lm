"""Descriptive (no rule, no new runs): the dev curves and clause statistics of every existing TM row at T = clauses,
per dataset, from results/explore/abcd.jsonl. Read only; standard library.

    python3 -I curves.py <study root> > curves.out
"""

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
rows = [json.loads(l) for l in (root / "results" / "explore" / "abcd.jsonl").read_text().splitlines()]
by = defaultdict(list)
for r in rows:
    if r.get("arm") != "tm" or not r.get("history") or "smoke" in r.get("tag", ""):
        continue
    c = r["cfg"]
    if c["T"] != c["n_clauses"] or c["n_clauses"] < 4000 or c["epochs"] != 40:
        continue
    by[r.get("dataset", "abcd")].append(r)  # rows from before 2026-10-17 have no "dataset" field: ABCD
print("rows used (T = clauses, >= 4000 clauses, 40 epochs, no smoke):", {k: len(v) for k, v in sorted(by.items())})
for ds, rs in sorted(by.items()):
    gaps, early, falls, med_lit, meanpos, empty_share, nopos_share = [], 0, [], [], [], [], []
    curve = defaultdict(list)
    for r in rs:
        h = [x["val_acc"] for x in r["history"]]
        ep = [x["epoch"] for x in r["history"]]
        b = max(h)
        gaps.append(100 * (b - sum(h[-10:]) / len(h[-10:])))
        falls.append(100 * (b - h[-1]))
        early += r["best_epoch"] <= 10
        for e, v in zip(ep, h):
            curve[e].append(100 * (v - b))
        s = r.get("stats", {})
        med_lit.append(s.get("median_literals"))
        meanpos.append(s.get("mean_positive_literals"))
        empty_share.append(100 * s.get("empty", 0) / r["cfg"]["n_clauses"])
        nopos_share.append(100 * s.get("no_positive_literal", 0) / r["cfg"]["n_clauses"])
    n = len(rs)
    print(f"\n{ds}: {n} rows; best epoch <= 10 in {early} of {n}; best-dev minus last-10 mean: median {st.median(gaps):.2f} "
          f"(min {min(gaps):.2f}, max {max(gaps):.2f}); best-dev minus final epoch: median {st.median(falls):.2f}")
    print(f"  clause median literals: median {st.median(med_lit)}; mean positive literals: median {st.median(meanpos):.2f}; "
          f"empty clauses: median {st.median(empty_share):.1f}%; clauses with no positive literal: median {st.median(nopos_share):.1f}%")
    print("  mean of (dev acc minus the row's best), by epoch: " + " ".join(f"{e}:{st.mean(v):+.2f}" for e, v in sorted(curve.items())))

# Part 2: the MultiWOZ rows by s (every T = clauses row of the s screens and references), with their clause shape.
print("\nmultiwoz rows by s (T = clauses): tag, seed, clauses, s, best-dev, last-10 mean, best epoch, median literals, "
      "mean positive literals, empty, no positive literal")
for r in sorted(by["multiwoz"], key=lambda r: (r["cfg"]["n_clauses"], r["cfg"]["s"], r["tag"], r["seed"])):
    h = [x["val_acc"] for x in r["history"]]
    s = r.get("stats", {})
    print(f"  {r['tag']:32s} {r['seed']} {r['cfg']['n_clauses']:>5} s {r['cfg']['s']:>4} {r['acc']:.4f} {sum(h[-10:]) / len(h[-10:]):.4f} "
          f"ep {r['best_epoch']:>2} med {s.get('median_literals')} pos {s.get('mean_positive_literals'):.2f} empty {s.get('empty')} "
          f"nopos {s.get('no_positive_literal')}")
