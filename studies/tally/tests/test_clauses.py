"""Compiled clause evaluation must equal tmu's predict-path clause outputs.

Run: PYTHONPATH=src python tests/test_clauses.py
Trains a small FastCoalesced bank on real T0 inputs (train split, 20,000 positions) and on
random one-hot inputs, and compares outputs on held-in and held-out rows.
"""
import numpy as np

from tally.t0 import data
from tally.t0.clauses import Bank
from tally.t0.fast_coalesced import FastCoalesced


def main():
    d = data.build()
    codes, y = data.split_xy(d, "train")
    F = data.CONTEXT * (data.K_TOP + 1)
    active = (codes + np.arange(data.CONTEXT) * (data.K_TOP + 1)).astype(np.int32)
    rows = np.random.default_rng(1).choice(len(y), 20000, replace=False)
    tm = FastCoalesced(300, 300, 5.0, platform="CPU", weighted_clauses=True, seed=3)
    tm.fit(data.to_bits(codes, rows), y[rows].astype(np.uint32))
    # tm.clause_outputs takes bits; Bank takes active indices
    for label, r in (("held-in", rows[:3000]), ("held-out", np.setdiff1d(np.arange(len(y)), rows)[:3000])):
        lib = tm.clause_outputs(data.to_bits(codes, r))
        bank = Bank(tm.clause_bank.get_literals(), F)
        ours = bank.fired(active[r])
        eq = np.array_equal(lib, ours)
        yy = y[r]
        N, nk = bank.counts(active[r], yy, data.K_TOP + 1)
        dense = np.zeros_like(N)
        np.add.at(dense.T, yy, ours.astype(np.int64))
        print(f"{label}: equal={eq} counts_equal={np.array_equal(N, dense)} mean_fired={ours.sum(1).mean():.1f} "
              f"empty={int((bank.nlit == 0).sum())} no_positive={len(bank.free)}")
        assert eq and np.array_equal(N, dense)
    rb = Bank.random_like(bank, np.random.default_rng(0))
    assert np.array_equal(rb.npos, bank.npos) and np.array_equal(rb.nlit, bank.nlit)
    print("PASS")


if __name__ == "__main__":
    main()
