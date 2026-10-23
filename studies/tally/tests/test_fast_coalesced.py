"""FastCoalesced must train to exactly the state of tmu's TMCoalescedClassifier.

Run: PYTHONPATH=src python tests/test_fast_coalesced.py [n_examples] [classes]
Trains both on the same random one-hot inputs with the same seed and compares automaton
states, weights, clause outputs and class sums for equality. Also checks that two runs of the
library itself agree (otherwise equality would mean nothing).
"""
import sys

import numpy as np
from tmu.models.classification.coalesced_classifier import TMCoalescedClassifier

from tally.t0.fast_coalesced import FastCoalesced


def data(n, K, F=600, ones=3, seed=0):
    rng = np.random.default_rng(seed)
    X = np.zeros((n, F), dtype=np.uint32)
    for k in range(ones):
        X[np.arange(n), rng.integers(k * F // ones, (k + 1) * F // ones, n)] = 1
    y = ((X[:, : F // ones].argmax(1) * 7 + rng.integers(0, 3, n)) % K).astype(np.uint32)
    y[:K] = np.arange(K)
    return X, y


def train(cls, X, y, K, epochs=2):
    tm = cls(200, 200, 10.0, platform="CPU", weighted_clauses=True, seed=7)
    tm.fit(X[:K], y[:K])
    for _ in range(epochs):
        tm.fit(X[K:], y[K:])
    return tm


def state(tm):
    W = np.stack([tm.weight_banks[i].get_weights() for i in range(tm.number_of_classes)])
    return tm.clause_bank.clause_bank.copy(), W.copy()


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    X, y = data(n, K)
    a, b, f = train(TMCoalescedClassifier, X, y, K), train(TMCoalescedClassifier, X, y, K), train(FastCoalesced, X, y, K)
    sa, sb, sf = state(a), state(b), state(f)
    lib_repeat = all(np.array_equal(p, q) for p, q in zip(sa, sb))
    fast_equal = all(np.array_equal(p, q) for p, q in zip(sa, sf))
    _, cs_lib = a.predict(X[:300], return_class_sums=True)
    co = f.clause_outputs(X[:300]).astype(np.int64)
    cs_fast = co @ f.weights().T.astype(np.int64)
    sums_equal = np.array_equal(cs_lib, cs_fast)
    included = int((sa[0] >= 128).sum())
    print(f"n={n} K={K} library_repeatable={lib_repeat} fast_equal={fast_equal} class_sums_equal={sums_equal} "
          f"weights_nonzero={int((sa[1] != 0).sum())} ta_bytes={sa[0].size}")
    assert lib_repeat and fast_equal and sums_equal
    print("PASS")


if __name__ == "__main__":
    main()
