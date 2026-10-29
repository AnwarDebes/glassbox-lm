#!/usr/bin/env python
"""N8 (rules in LOG.md before any run): job lists and selection, mechanical.

    n8_select.py s1            # print the stage-1 jobs (TREC and MR, bow, seed 1: negation x T x s)
    n8_select.py s2            # from the stage-1 rows: the best setting per set, then its 4 one-change variants
    n8_select.py s3            # from the stage-1 and stage-2 rows: the selected setting per set, seeds 2-3 bow and
                               # seeds 1-3 bow+existsT_potion
    n8_select.py words         # the words of the rule (CAP / PARTLY / CEILING) from the stage-3 rows

Selection: the highest S1-val (val_mean_last50) of view bow, seed 1; ties: the earlier setting in the listed order
(stage 1 in the order of its job list; stage 2: the stage-1 best first, then the variants in the listed order).
A job line: dataset|stage|view|seed|clauses|T|s|negation|batch|neg|max_literals.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SETS = ("trec", "mr")
LR = {"trec": 86.422, "mr": 72.113}  # N7-F1 diagnostic, logistic regression on the same bits (carved val)
KEYS = ("clauses", "T", "s", "negation", "batch", "neg", "max_literals")


def s1_settings():
    out = []
    for neg in (1, 0):
        for T in (80, 1000, 4000, 8000):
            for s in (2.0, 5.0, 9.0, 15.0):
                out.append({"clauses": 4000, "T": T, "s": s, "negation": neg, "batch": 128, "neg": "hardest",
                            "max_literals": 0})
    return out


def variants(best):
    out = []
    for change in ({"clauses": 10000}, {"batch": 32}, {"neg": "random"}, {"max_literals": 32}):
        v = dict(best)
        v.update(change)
        out.append(v)
    return out


def line(ds, stage, view, seed, st):
    return "|".join([ds, stage, view, str(seed)] + [str(st[k]) for k in KEYS])


def rows():
    p = HERE / "results_n8_sweep.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def same(a, b):
    return all(str(a[k]) == str(b[k]) for k in KEYS)


def score(R, ds, st, view="bow", seed=1):
    hits = [r for r in R if r["dataset"] == ds and r["view"] == view and int(r["seed"]) == seed and same(r["setting"], st)]
    return hits[0]["val_mean_last50"] if hits else None


def best_of(R, ds, cands, allow_missing=False):
    # allow_missing (stage-2 variants only): a variant without a row (a failed run, e.g. out of memory) is left out
    # and named on stderr; the stage-1 best (the first candidate) must always have a row
    sc = [(score(R, ds, st), i, st) for i, st in enumerate(cands)]
    missing = [st for v, _, st in sc if v is None]
    if missing and (not allow_missing or sc[0][0] is None):
        sys.exit(f"missing rows for {ds}: {missing}")
    for st in missing:
        print(f"# {ds}: no row for {st}; left out", file=sys.stderr)
    v, _, st = max((t for t in sc if t[0] is not None), key=lambda t: (t[0], -t[1]))
    return st, v


def selected(R, ds):
    b1, _ = best_of(R, ds, s1_settings())
    return best_of(R, ds, [b1] + variants(b1), allow_missing=True)


def main() -> None:
    what = sys.argv[1]
    R = rows()
    if what == "s1":
        for ds in SETS:
            for st in s1_settings():
                print(line(ds, "s1", "bow", 1, st))
    elif what == "s2":
        for ds in SETS:
            b1, v = best_of(R, ds, s1_settings())
            print(f"# {ds} stage-1 best {b1} S1-val {v}", file=sys.stderr)
            for st in variants(b1):
                print(line(ds, "s2", "bow", 1, st))
    elif what == "s3":
        for ds in SETS:
            st, v = selected(R, ds)
            print(f"# {ds} selected {st} S1-val {v}", file=sys.stderr)
            for seed in (2, 3):
                print(line(ds, "s3", "bow", seed, st))
            for seed in (1, 2, 3):
                print(line(ds, "s3", "bow+existsT_potion", seed, st))
    elif what == "words":
        n7 = [json.loads(x) for x in (HERE / "results_n7_test.jsonl").read_text().splitlines() if x.strip()]
        cap = part = 0
        for ds in SETS:
            st, v1 = selected(R, ds)
            fresh = [score(R, ds, st, "bow", s) for s in (2, 3)]
            if any(x is None for x in fresh):
                sys.exit(f"stage 3 incomplete for {ds}")
            ref = [r["val_mean_last50"] for r in n7 if r["dataset"] == ds and r["view"] == "bow" and int(r["seed"]) in (2, 3)]
            m, mr = float(np.mean(fresh)), float(np.mean(ref))
            cap += m >= LR[ds] - 2.0 - 1e-9
            part += m >= mr + 5.0 - 1e-9
            ex = [score(R, ds, st, "bow+existsT_potion", s) for s in (1, 2, 3)]
            exr = [r["val_mean_last50"] for r in n7 if r["dataset"] == ds and r["view"] == "bow+existsT_potion" and int(r["seed"]) in (1, 2, 3)]
            print(f"{ds}: selected {st}; bow S1-val seed 1 {v1} (selection), seeds 2-3 {fresh} mean {m:.2f}; "
                  f"N7 setting seeds 2-3 mean {mr:.2f} ({m - mr:+.2f}); logistic regression {LR[ds]} ({m - LR[ds]:+.2f}); "
                  f"bow+existsT_potion seeds 1-3 {ex} (N7 setting {exr})")
        word = "CAP" if cap == len(SETS) else ("PARTLY" if part == len(SETS) else "CEILING")
        print(f"words: within 2.0 of logistic regression on {cap} of 2, +5.0 over N7's setting on {part} of 2 -> {word}")
    else:
        sys.exit(f"unknown: {what}")


if __name__ == "__main__":
    main()
