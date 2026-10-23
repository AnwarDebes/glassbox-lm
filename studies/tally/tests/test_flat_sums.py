"""Class sums of the flat TM from the compiled clause bank must equal tmu's predict class sums.

Run: PYTHONPATH=src python tests/test_flat_sums.py
"""
import numpy as np
from tmu.models.classification.vanilla_classifier import TMClassifier

from tally.t0 import data, readout
from tally.t0.run import actives, machine_bank


def main():
    d = data.build()
    codes, y = data.split_xy(d, "train")
    rows = np.random.default_rng(2).choice(len(y), 20000, replace=False)
    tm = TMClassifier(10, 10, 10.0, platform="CPU", seed=5)
    tm.fit(data.to_bits(codes, rows), y[rows].astype(np.uint32))
    test = np.random.default_rng(3).choice(len(y), 300, replace=False)
    _, lib = tm.predict(data.to_bits(codes, test), return_class_sums=True)
    bank, Ml = machine_bank(tm, "flat")
    act = actives(codes[test])
    ours = readout.scores(lambda lo, hi: bank.fired(act[lo:hi]), Ml, len(test))
    eq = np.array_equal(lib.astype(np.float64), ours)
    print(f"flat class sums equal={eq} clauses={bank.C} mean_fired={bank.fired(act).sum(1).mean():.1f}")
    assert eq
    print("PASS")


if __name__ == "__main__":
    main()
