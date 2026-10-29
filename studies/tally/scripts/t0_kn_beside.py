"""T0, reported beside and not compared: Kneser-Ney 3-gram and 5-gram on dev-test.

    PYTHONPATH=src python scripts/t0_kn_beside.py [--split devtest|valid]

The registration of T0 (registrations/T0.md, section "Arms") reports interpolated modified
Kneser-Ney 3-gram and 5-gram models on the same splits beside the five arms, with no comparison
and no rule. This script was written on 2026-10-25, after the freeze of T0 and before its verdict
was computed. It uses the code of step A1 (src/tally/floor/kn.py, which
scripts/a1_baseline_check.py checks against KenLM on valid) in the same way: counts from the
train split only, sentence mode, full distribution over V = 10,000. It scores the dev-test split,
runs the section-4 contract on 1,000 random dev-test positions (the vector over V sums to 1, its
minimum is above 0 and the fast path equals the vector entry), and stores the per-position bits
(bits.npz, keys kn3__devtest and kn5__devtest, with blocks__devtest) in its run folder.
--split valid is the smoke test: on valid the script must reproduce the KN3 and KN5 bits per token
that scripts/a1_baseline_check.py recorded (valid_table, own: KN3 and own: KN5).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from tally import ptb  # noqa: E402
from tally.floor import stats  # noqa: E402
from tally.floor.kn import KN  # noqa: E402
from tally.runlog import Run  # noqa: E402

ORDERS = (3, 5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("devtest", "valid"), default="devtest")
    SPLIT = ap.parse_args().split
    cfg = {"orders": list(ORDERS), "mode": "sent", "counts_from": "train", "split": SPLIT,
           "contract_positions": 1000, "role": "T0 reported beside, not compared"}
    run = Run(f"t0_kn_beside_{SPLIT}", cfg, data_files=[ptb.path("train"), ptb.path("valid")])
    vocab, w2i, ids, blocks = ptb.load()
    V, eos = len(vocab), w2i["<eos>"]
    out, metrics = {f"blocks__{SPLIT}": blocks[SPLIT]}, {"tokens": int(len(ids[SPLIT])), "V": V}
    for N in ORDERS:
        m = KN(ids["train"], V, eos, order=N, mode="sent")
        lp = m.logprob(ids[SPLIT])
        assert len(lp) == len(ids[SPLIT]), (len(lp), len(ids[SPLIT]))
        r = stats.contract(lambda pos: m.dists(ids[SPLIT], pos), lp, len(ids[SPLIT]), 1000, seed=N,
                           true_ids=ids[SPLIT])
        run.print(f"contract kn{N} {SPLIT}", r)
        assert r["pass"], f"section-4 contract failed for kn{N} on {SPLIT}: {r}"
        bits = -lp / np.log(2.0)
        out[f"kn{N}__{SPLIT}"] = bits
        metrics[f"kn{N}_{SPLIT}_bpw"] = float(bits.mean())
        metrics[f"kn{N}_{SPLIT}_ppl"] = float(2 ** bits.mean())
        metrics[f"contract_kn{N}_{SPLIT}"] = r
        metrics[f"kn{N}_summary"] = m.summary()
    np.savez_compressed(run.path("bits.npz"), **out)
    run.finish(metrics)


if __name__ == "__main__":
    main()
