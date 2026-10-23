"""T0 verdict from the final runs (registration registrations/T0.md).

    python -m tally.t0.report <final run folder>... --logreg <logreg run folder>...

Per position of devtest, each arm's bits are averaged over the seeds; the paired difference
(counted minus learned, and the other comparisons) is resampled by blocks of 100 sentences
(1,000 resamples, percentile 95% interval). Prints and writes report.json in its own run folder.
"""
import argparse
import json
import os

import numpy as np

from tally import runlog

B = 1000


def block_ci(diff, blocks, rng):
    ub, inv = np.unique(blocks, return_inverse=True)
    sums = np.bincount(inv, weights=diff)
    cnts = np.bincount(inv)
    stats = []
    for _ in range(B):
        pick = rng.integers(0, len(ub), len(ub))
        stats.append(sums[pick].sum() / cnts[pick].sum())
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


def verdict(mean, lo, hi, a, b):
    if hi < 0:
        return f"{a} beats {b}"
    if lo > 0:
        return f"{b} beats {a}"
    return "no difference shown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--logreg", nargs="*", default=[])
    ap.add_argument("--flat", nargs="*", default=[])
    ap.add_argument("--split", default="devtest")
    args = ap.parse_args()
    sp = args.split
    run = runlog.Run(f"t0_report_{sp}", vars(args))
    rng = np.random.default_rng(0)

    def stack(folders, key):
        arrs = []
        for f in folders:
            z = np.load(os.path.join(f, "bits.npz"))
            arrs.append(z[f"{key}__{sp}"])
        return np.mean(arrs, 0), len(arrs)

    z0 = np.load(os.path.join(args.runs[0], "bits.npz"))
    blocks = z0[f"blocks__{sp}"]
    arms = {}
    arms["learned"], n_seeds = stack(args.runs, "learned")
    arms["counted"], _ = stack(args.runs, "counted")
    unt = [k.split("__")[0] for k in z0.files if k.startswith("untrained") and k.endswith(f"__{sp}")]
    arms["untrained"] = np.mean([stack(args.runs, k)[0] for k in unt], 0)
    if args.logreg:
        arms["logreg"], _ = stack(args.logreg, "logreg")
    if args.flat:
        arms["flat_learned"], _ = stack(args.flat, "learned")
        arms["flat_counted"], _ = stack(args.flat, "counted")
    rep = dict(split=sp, seeds=n_seeds, positions=int(len(blocks)), blocks=int(len(np.unique(blocks))),
               bpw={k: float(v.mean()) for k, v in arms.items()},
               ppl={k: float(2 ** v.mean()) for k, v in arms.items()})
    pairs = [("counted", "learned"), ("counted", "untrained"), ("counted", "logreg"),
             ("flat_counted", "flat_learned")]
    for a, b in pairs:
        if a in arms and b in arms:
            m, lo, hi = block_ci(arms[a] - arms[b], blocks, rng)
            rep[f"{a}_minus_{b}"] = dict(mean=m, ci95=[lo, hi], reading=verdict(m, lo, hi, a, b))
    rep["primary"] = rep["counted_minus_learned"]["reading"]
    m, lo, hi = block_ci(arms["counted"] - arms["untrained"], blocks, rng)
    rep["tsetlin_learning_helps_counted_readout"] = bool(hi < 0)
    run.finish(rep)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
