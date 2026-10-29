#!/usr/bin/env python
"""N9 (EXPLORATORY, carved val only; plan LOG.md 12:49:05Z, rules in LOG.md before any run): job lists, selection and
words, mechanical. n8_select.py (83c72196fec0ae2a) in substance: the same stage-1 grid, the same stage-2 variants and
the same selection rule (the highest S1-val = val_mean_last50 of seed 1; ties: the earlier setting in the listed
order; a stage-2 variant without a row is left out and named on stderr), applied per view.

    n9_select.py check     # the check job: TREC bow seed 1 at N7's setting (the chain runs it with --check-n8)
    n9_select.py n9s0      # bow arm: N8's stage-2 variants of the bow stage-1 best that have no row (TREC's lost run)
    n9_select.py n9s1      # combined arm (bow+existsT_potion): the 32 stage-1 settings, seed 1
    n9_select.py n9s2      # combined arm: the 4 one-change variants of its stage-1 best, seed 1
    n9_select.py n9s3      # numbers, seeds 2-4: bow at its pick, bow+existsT_potion at its own pick, and
                           # bow+existsT_glove and bow+existsT_c4 at the potion view's pick (declared)
    n9_select.py words     # per set and table: adds / does not add, and the reading fixed in the plan

Rows: results_n8_sweep.jsonl first, then results_n9_sweep.jsonl. A run that already has a row for the same set, view,
seed and setting is not listed again (its first row is used; the check job is the one exception).
"Adds" (per set and table): the mean over seeds 2-4 of (combined - bow, paired by seed, S1-val) is > 0 and all 3 paired
differences are > 0.
A job line: dataset|stage|view|seed|clauses|T|s|negation|batch|neg|max_literals.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SETS = ("trec", "mr")
KEYS = ("clauses", "T", "s", "negation", "batch", "neg", "max_literals")
BOW, COMB = "bow", "bow+existsT_potion"
TABLES = (COMB, "bow+existsT_glove", "bow+existsT_c4")
NAME = {COMB: "potion (D_potion)", "bow+existsT_glove": "GloVe 42B (D_glove)",
        "bow+existsT_c4": "C4 words ppw_S1000 (W1-T 'adds over bow')"}
SEEDS = (2, 3, 4)
N7SET = {"clauses": 4000, "T": 80, "s": 9.0, "negation": 1, "batch": 128, "neg": "hardest", "max_literals": 0}
READING = {2: "also with both views tuned (carved val, TREC and MR)",
           0: "measured at a TM setting tuned on neither view; with both views tuned the meaning bits did not add on "
              "carved val (TREC, MR)",
           1: "mixed (adds on 1 of 2 sets with both views tuned), said so"}


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


def rows(only_n8=False):
    out = []
    for name in ("results_n8_sweep.jsonl",) + (() if only_n8 else ("results_n9_sweep.jsonl",)):
        p = HERE / name
        out += [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
    return out


def same(a, b):
    return all(str(a[k]) == str(b[k]) for k in KEYS)


def row(R, ds, st, view, seed):
    hits = [r for r in R if r["dataset"] == ds and r["view"] == view and int(r["seed"]) == seed and same(r["setting"], st)]
    return hits[0] if hits else None


def score(R, ds, st, view, seed=1):
    r = row(R, ds, st, view, seed)
    return r["val_mean_last50"] if r else None


def best_of(R, ds, cands, view, allow_missing=False):
    # allow_missing (stage-2 variants only): a variant without a row (a failed run, e.g. out of memory) is left out
    # and named on stderr; the stage-1 best (the first candidate) must always have a row
    sc = [(score(R, ds, st, view), i, st) for i, st in enumerate(cands)]
    missing = [st for v, _, st in sc if v is None]
    if missing and (not allow_missing or sc[0][0] is None):
        sys.exit(f"missing rows for {ds} {view}: {missing}")
    for st in missing:
        print(f"# {ds} {view}: no row for {st}; left out", file=sys.stderr)
    v, _, st = max((t for t in sc if t[0] is not None), key=lambda t: (t[0], -t[1]))
    return st, v


def pick(R, ds, view):
    b1, _ = best_of(R, ds, s1_settings(), view)
    return best_of(R, ds, [b1] + variants(b1), view, allow_missing=True)


def todo(R, ds, stage, view, seed, st):
    return [] if row(R, ds, st, view, seed) else [line(ds, stage, view, seed, st)]


def words(R) -> list:
    out, adds = [], {t: 0 for t in TABLES}
    R8 = rows(only_n8=True)
    for ds in SETS:
        pb, vb = pick(R, ds, BOW)
        p8, _ = pick(R8, ds, BOW)
        pc, vc = pick(R, ds, COMB)
        out.append(f"{ds}: bow pick {pb} (seed-1 S1-val {vb}; N8's pick {'the same' if same(pb, p8) else p8}); "
                   f"bow+existsT_potion pick {pc} (seed-1 S1-val {vc})")
        rb = [row(R, ds, pb, BOW, s) for s in SEEDS]
        for t in TABLES:
            rt = [row(R, ds, pc, t, s) for s in SEEDS]
            if any(r is None for r in rb + rt):
                gone = [f"{v} seed {s}" for v, rr in ((BOW, rb), (t, rt)) for s, r in zip(SEEDS, rr) if r is None]
                out.append(f"  {NAME[t]}: incomplete, no row for {gone}")
                continue
            a = [r["val_mean_last50"] for r in rt]
            b = [r["val_mean_last50"] for r in rb]
            d = [round(x - y, 3) for x, y in zip(a, b)]
            ok = float(np.mean(d)) > 0 and all(x > 0 for x in d)
            adds[t] += ok
            out.append(f"  {NAME[t]}: seeds 2-4 S1-val {a} vs bow {b}; paired differences {d}, mean {np.mean(d):+.2f} "
                       f"-> {'adds' if ok else 'does not add'} (best-epoch val {[r['val_at_best'] for r in rt]} vs "
                       f"{[r['val_at_best'] for r in rb]})")
        n8c = [score(R8, ds, p8, COMB, s) for s in (1, 2, 3)]
        out.append(f"  descriptive: N8's bow+existsT_potion rows at N8's bow pick, seeds 1-3: {n8c}")
    for t in TABLES:
        out.append(f"words {NAME[t]}: adds on {adds[t]} of 2 sets -> {READING[adds[t]]}")
    return out


def main() -> None:
    what = sys.argv[1]
    R = rows()
    if what == "check":
        print(line("trec", "n9check", BOW, 1, N7SET))
    elif what == "n9s0":
        for ds in SETS:
            b1, v = best_of(R, ds, s1_settings(), BOW)
            for st in variants(b1):
                for j in todo(R, ds, "n9s0", BOW, 1, st):
                    print(j)
    elif what == "n9s1":
        for ds in SETS:
            for st in s1_settings():
                for j in todo(R, ds, "n9s1", COMB, 1, st):
                    print(j)
    elif what == "n9s2":
        for ds in SETS:
            b1, v = best_of(R, ds, s1_settings(), COMB)
            print(f"# {ds} {COMB} stage-1 best {b1} S1-val {v}", file=sys.stderr)
            for st in variants(b1):
                for j in todo(R, ds, "n9s2", COMB, 1, st):
                    print(j)
    elif what == "n9s3":
        for ds in SETS:
            pb, vb = pick(R, ds, BOW)
            pc, vc = pick(R, ds, COMB)
            print(f"# {ds} bow pick {pb} S1-val {vb}; {COMB} pick {pc} S1-val {vc}", file=sys.stderr)
            for seed in SEEDS:
                for view, st in ((BOW, pb),) + tuple((t, pc) for t in TABLES):
                    for j in todo(R, ds, "n9s3", view, seed, st):
                        print(j)
    elif what == "words":
        print("\n".join(words(R)))
    else:
        sys.exit(f"unknown: {what}")


if __name__ == "__main__":
    main()
