#!/usr/bin/env python
"""Verdict of REGISTRATION_W1T.md (written and self-tested on synthetic rows before any test read).

    verdict_w1t.py [--word FILE] [--n7 FILE]

Arms per set (TREC-6, MR, SST-2, R52, R8; seeds 1-5): the C4-word table (rows of w1/results_w1t_test.jsonl, view
existsT_c4words) and N7's registered test rows (results_n7_test.jsonl, views existsT_potion, existsT_glove, bow). A
set counts for a comparison only when both arms have all 5 seeds; a repeated (set, view, seed) row is ignored (the
first is kept). Statistics: S2 = test accuracy at the best carved-val epoch (the primary, a single read), S1 = mean
test accuracy over epochs 51-100 (beside it). Paired by seed:
  word - potion   per set the mean of the 5 paired differences on S2; "within 1.0" when >= -1.0, "within 2.0" when
                  >= -2.0. REACHES potion = within 1.0 on >= 3 of 5 sets; NEAR = within 2.0 on >= 3 of 5; else BELOW.
  word - GloVe 42B the same words.
  word - bow      N7's D rule: on a set "adds" when the mean paired S2 difference is > 0 and >= 4 of the 5 seed
                  differences are > 0; met when it adds on >= 3 of 5 sets ("adds on N of 5 sets", never "significant").
Comparisons allow 1e-9.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SETS = ("trec", "mr", "sst2", "r52", "r8")
SEEDS = (1, 2, 3, 4, 5)
EPS = 1e-9


def rows(path: Path, reg: str, views) -> dict:
    out, dups = {}, []
    if not path.exists():
        return out
    for x in path.read_text().splitlines():
        if not x.strip():
            continue
        r = json.loads(x)
        if r.get("registration") != reg or r.get("mode") != "test" or r.get("view") not in views:
            continue
        k = (r["dataset"], r["view"], int(r["seed"]))
        if k in out:
            dups.append(k)
        else:
            out[k] = r
    if dups:
        print(f"repeated rows ignored ({path.name}): {dups}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--word", type=Path, default=HERE / "results_w1t_test.jsonl")
    ap.add_argument("--n7", type=Path, default=HERE.parent / "results_n7_test.jsonl")
    args = ap.parse_args()
    R = {**rows(args.word, "REGISTRATION_W1T.md", ("existsT_c4words",)),
         **rows(args.n7, "REGISTRATION_N7.md", ("existsT_potion", "existsT_glove", "bow"))}
    print(f"REGISTRATION_W1T.md verdict: {len(R)} test rows")

    def vals(ds, v, key):
        a = [R[(ds, v, s)][key] for s in SEEDS if (ds, v, s) in R]
        return np.array(a) if len(a) == len(SEEDS) else None

    for ds in SETS:
        line = f"== {ds}:"
        for v in ("existsT_c4words", "existsT_potion", "existsT_glove", "bow"):
            s2, s1 = vals(ds, v, "test_at_best_val"), vals(ds, v, "test_mean_last50")
            line += f" | {v} S2 {s2.mean():.2f} +- {s2.std(ddof=1):.2f} S1 {s1.mean():.2f}" if s2 is not None else f" | {v} incomplete"
        print(line)

    def compare(ref: str, name: str):
        w1 = w2 = 0
        missing = []
        for ds in SETS:
            a, b = vals(ds, "existsT_c4words", "test_at_best_val"), vals(ds, ref, "test_at_best_val")
            a1, b1 = vals(ds, "existsT_c4words", "test_mean_last50"), vals(ds, ref, "test_mean_last50")
            if a is None or b is None:
                missing.append(ds)
                continue
            d, d1 = a - b, a1 - b1
            w1 += d.mean() >= -1.0 - EPS
            w2 += d.mean() >= -2.0 - EPS
            print(f"   word - {name} {ds}: S2 {d.mean():+.2f} (seeds > 0: {int((d > EPS).sum())} of 5) | S1 {d1.mean():+.2f}")
        word = "REACHES" if w1 >= 3 else ("NEAR" if w2 >= 3 else "BELOW")
        inc = f" INCOMPLETE (missing {', '.join(missing)})" if missing else ""
        print(f"   word vs {name}: within 1.0 on {w1} of 5, within 2.0 on {w2} of 5 -> {word} {name.upper()}{inc}")

    print("\n== word control against potion (primary) and against GloVe 42B, S2 paired by seed")
    compare("existsT_potion", "potion")
    compare("existsT_glove", "GloVe 42B")
    print("\n== word control against bow (N7's D rule, S2 paired by seed)")
    n_add, missing = 0, []
    for ds in SETS:
        a, b = vals(ds, "existsT_c4words", "test_at_best_val"), vals(ds, "bow", "test_at_best_val")
        if a is None or b is None:
            missing.append(ds)
            continue
        d = a - b
        adds = d.mean() > EPS and int((d > EPS).sum()) >= 4
        n_add += adds
        print(f"   word - bow {ds}: S2 {d.mean():+.2f} (seeds > 0: {int((d > EPS).sum())} of 5) -> {'adds' if adds else 'does not add'}")
    print(f"   adds on {n_add} of 5 sets -> {'MET' if n_add >= 3 else 'NOT MET'}" + (f" INCOMPLETE (missing {', '.join(missing)})" if missing else ""))


if __name__ == "__main__":
    main()
