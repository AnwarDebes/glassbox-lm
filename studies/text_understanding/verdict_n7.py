#!/usr/bin/env python
"""Verdict of REGISTRATION_N7.md: reads results_n7_test.jsonl (or --results for the self-test).

    verdict_n7.py [--results F]

Per dataset and view, over seeds 1-5: S1 = Yadav et al.'s statistic (the mean test accuracy over epochs 51-100) and
S2 = the single-read number (test accuracy at the best epoch on our carved val split), mean and sd (ddof 1). A view
counts on a dataset only with all 5 seeds; a repeated (dataset, view, seed) row is ignored (the first is kept).
Bars (S1 means; written before any test read):
  A  calibration: our bag-of-words TM within 2.0 of Yadav's bag-of-words TM on at least 3 of TREC-6, MR, R52, R8.
  B  same table: bow+existsT_glove above Yadav's best TM+GloVe row on at least 3 of the 4.
  C  potion: bow+existsT_potion above Yadav's best TM+GloVe row on at least 3 of the 4; strong: also above the best
     published TM number with a stated averaging protocol of each dataset on at least 3 of the 4 (MR 77.51 and R8
     97.50 are Yadav's k rows, k chosen on the test set: optimistic published numbers).
  D  meaning bits add to words (internal, same setting, splits and seeds; stands even if A fails), on S2 (the single
     read at the best val epoch), with S1 printed beside it: on a set, bow+existsT_X - bow "adds" when the mean over
     the 5 seeds is > 0 and at least 4 of the 5 paired seed differences are > 0; D_potion and D_glove are met when it
     adds on at least 3 of the 5 sets. This is a weak per-set test: write "adds on N of 5 sets", never "significant".
  E  table: bow+existsT_potion - bow+existsT_glove, the same paired rule on S2, met on at least 3 of the 5 sets.
A is two-sided (the direction is printed per set). SST-2 is reported against the Sparse TM's average (no bar for
the published comparison: its split is not stated; it does count for D and E). Comparisons allow 1e-9.
"""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEEDS = (1, 2, 3, 4, 5)
DATASETS = ("trec", "mr", "r52", "r8", "sst2")
VIEWS = ("bow", "existsT_potion", "existsT_glove", "bow+existsT_potion", "bow+existsT_glove")
BARS4 = ("trec", "mr", "r52", "r8")
# Yadav et al. 2021 (BlackboxNLP; Table 4 for TREC, Table 3 for R8 / R52 / MR; mean of the last 50 epochs x 5 runs)
YADAV_BOW = {"trec": 88.05, "mr": 75.14, "r52": 84.62, "r8": 96.16}
YADAV_GLOVE = {"trec": 90.04, "mr": 77.51, "r52": 89.14, "r8": 97.50}  # best of "TM with k" / "TM with phi"
# best published TM number with a stated averaging protocol (no neural network): Contracting TM (Bhattarai et al.
# 2023, Table I, last-25 average) for TREC 91.61 and R52 89.53; Yadav for MR 77.51 and R8 97.50
BEST_CLEAN_TM = {"trec": 91.61, "mr": 77.51, "r52": 89.53, "r8": 97.50}
SPARSE_TM_SST2 = 78.93  # Østby et al. 2024, Table I, average of the last 25 epochs; split not stated
NON_TM = {"trec": "CNN-non-static 93.6 (Kim 2014 T2), SWEM-max 89.0 (Shen 2018 T8), BERT-base 97.20 (Sun 2019)",
          "mr": "CNN-non-static 77.75, Bi-LSTM 77.68 (TextGCN T2, same split), BERT 85.7 (BertGCN, same split)",
          "r52": "Bi-LSTM 90.54, TextGCN 93.56 (TextGCN T2), BERT 96.4 (BertGCN)",
          "r8": "Bi-LSTM 96.31, TextGCN 97.07 (TextGCN T2), BERT 97.8 (BertGCN)",
          "sst2": "CNN-non-static 87.2 (Kim 2014 T2), SWEM-max 83.6 (Shen 2018 T8), BERT-base 93.5 (Devlin 2019, GLUE test)"}
EPS = 1e-9


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=HERE / "results_n7_test.jsonl")
    args = ap.parse_args()
    res, dups = {}, []
    if args.results.exists():
        for x in args.results.read_text().splitlines():
            if x.strip():
                r = json.loads(x)
                if r.get("registration") != "REGISTRATION_N7.md" or r.get("mode") != "test":
                    continue
                k = (r["dataset"], r["view"], int(r["seed"]))
                (dups.append(k) if k in res else res.__setitem__(k, r))
    print(f"REGISTRATION_N7.md verdict: {len(res)} test rows from {args.results}" + (f"; repeated rows ignored: {dups}" if dups else ""))

    def stat(ds, v, key):
        a = [res[(ds, v, s)][key] for s in SEEDS if (ds, v, s) in res]
        return (float(np.mean(a)), float(np.std(a, ddof=1))) if len(a) == len(SEEDS) else None

    for ds in DATASETS:
        print(f"\n== {ds} (non-TM references: {NON_TM[ds]})")
        if ds in YADAV_BOW:
            print(f"   published TM: Yadav bag of words {YADAV_BOW[ds]}, Yadav TM+GloVe best {YADAV_GLOVE[ds]}, best stated-protocol TM {BEST_CLEAN_TM[ds]}")
        else:
            print(f"   published TM: Sparse TM {SPARSE_TM_SST2} (average of the last 25 epochs, split not stated)")
        for v in VIEWS:
            s1, s2 = stat(ds, v, "test_mean_last50"), stat(ds, v, "test_at_best_val")
            got = [res[(ds, v, s)] for s in SEEDS if (ds, v, s) in res]
            if not got:
                print(f"   {v:20s} no rows")
                continue
            on = got[0]["on_share"]
            line = (f"   {v:20s} S1 {s1[0]:6.2f} +- {s1[1]:.2f}" if s1 else f"   {v:20s} S1 incomplete ({len(got)} seeds)")
            line += f" | S2 {s2[0]:6.2f} +- {s2[1]:.2f}" if s2 else ""
            line += f" | bits {got[0]['n_bits']}, on share train {on['train']:.3f} test {on['test']:.3f}"
            line += f", best val epochs {[r['best_val_epoch'] for r in got]}"
            print(line)

    print("\n== bars (S1, mean of 5 seeds)")
    a = b = c = cs = 0
    miss = {"A": [], "B": [], "C": []}
    for ds in BARS4:
        bow, gl, po = stat(ds, "bow", "test_mean_last50"), stat(ds, "bow+existsT_glove", "test_mean_last50"), stat(ds, "bow+existsT_potion", "test_mean_last50")
        if bow:
            ok = abs(bow[0] - YADAV_BOW[ds]) <= 2.0 + EPS
            a += ok
            side = "above" if bow[0] > YADAV_BOW[ds] else "below"
            print(f"   A {ds}: our bag of words {bow[0]:.2f} vs Yadav's {YADAV_BOW[ds]} ({bow[0] - YADAV_BOW[ds]:+.2f}, {side}) -> {'within 2.0' if ok else 'OUTSIDE 2.0'}")
        else:
            miss["A"].append(ds)
        if gl:
            ok = gl[0] > YADAV_GLOVE[ds] + EPS
            b += ok
            print(f"   B {ds}: bow+existsT_glove {gl[0]:.2f} vs Yadav TM+GloVe {YADAV_GLOVE[ds]} ({gl[0] - YADAV_GLOVE[ds]:+.2f}) -> {'above' if ok else 'not above'}")
        else:
            miss["B"].append(ds)
        if po:
            ok, oks = po[0] > YADAV_GLOVE[ds] + EPS, po[0] > BEST_CLEAN_TM[ds] + EPS
            c, cs = c + ok, cs + oks
            print(f"   C {ds}: bow+existsT_potion {po[0]:.2f} vs Yadav TM+GloVe {YADAV_GLOVE[ds]} ({po[0] - YADAV_GLOVE[ds]:+.2f}), "
                  f"vs best stated-protocol TM {BEST_CLEAN_TM[ds]} ({po[0] - BEST_CLEAN_TM[ds]:+.2f})")
        else:
            miss["C"].append(ds)
    inc = lambda k: f" INCOMPLETE (missing {', '.join(miss[k])})" if miss[k] else ""  # noqa: E731
    print(f"   A calibration (within 2.0 on >= 3 of 4): {a} of 4 -> {'HOLDS' if a >= 3 else 'DOES NOT HOLD'}{inc('A')}")
    print(f"   B same table (above on >= 3 of 4): {b} of 4 -> {'MET' if b >= 3 else 'NOT MET'}{inc('B')}")
    print(f"   C potion (above on >= 3 of 4): {c} of 4 -> {'MET' if c >= 3 else 'NOT MET'}; strong (above best stated-protocol TM on >= 3 of 4): "
          f"{cs} of 4 -> {'MET' if c >= 3 and cs >= 3 else 'NOT MET'}{inc('C')}")
    def paired(ds, va, vb, key="test_at_best_val"):
        d = [res[(ds, va, k)][key] - res[(ds, vb, k)][key] for k in SEEDS if (ds, va, k) in res and (ds, vb, k) in res]
        return np.array(d) if len(d) == len(SEEDS) else None

    print("\n== internal bars (S2 = test at the best val epoch, paired over seeds 1-5; S1 beside it)")
    for name, va, vb in (("D_potion", "bow+existsT_potion", "bow"), ("D_glove", "bow+existsT_glove", "bow"),
                         ("E", "bow+existsT_potion", "bow+existsT_glove")):
        n_add, missing = 0, []
        for ds in DATASETS:
            d = paired(ds, va, vb)
            if d is None:
                missing.append(ds)
                continue
            adds = d.mean() > EPS and int((d > EPS).sum()) >= 4
            n_add += adds
            d1 = paired(ds, va, vb, "test_mean_last50")
            print(f"   {name} {ds}: {va} - {vb} S2 {d.mean():+.2f} (seeds > 0: {int((d > EPS).sum())} of 5) -> "
                  f"{'adds' if adds else 'does not add'}; S1 {d1.mean():+.2f} (seeds > 0: {int((d1 > EPS).sum())} of 5)")
        print(f"   {name}: adds on {n_add} of 5 sets -> {'MET' if n_add >= 3 else 'NOT MET'}" + (f" INCOMPLETE (missing {', '.join(missing)})" if missing else ""))
    s = stat("sst2", "bow+existsT_potion", "test_mean_last50"), stat("sst2", "bow+existsT_glove", "test_mean_last50")
    if all(s):
        print(f"   SST-2 (no bar): bow+existsT_potion {s[0][0]:.2f}, bow+existsT_glove {s[1][0]:.2f} vs Sparse TM {SPARSE_TM_SST2}")


if __name__ == "__main__":
    main()
