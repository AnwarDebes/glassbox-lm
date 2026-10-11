"""docs/11 section 6, kill criterion KB1 (docs/08): in dense-equivalent mode the oracle is a
standard Coalesced Tsetlin Machine and must reach accuracy parity with tmu on small tasks.

Protocol (fixed 2026-09-22 after measuring that BOTH implementations oscillate from epoch to
epoch on NoisyXOR, tmu between 0.87 and 1.00 and the oracle between 0.84 and 0.998, so a single
snapshot is noise): same task, same clauses, T, s and epoch count; the statistic is the MEAN
test accuracy over epochs 4 to 10 and over 3 runs; the band is 2 points (KB1).

Reference: tmu at commit 5605ff07 `TMCoalescedClassifier`, CPU, weighted clauses. Its CPU path is
unseeded C rand(), so its runs differ; the oracle is bit-exactly reproducible by seed. tmu with the
`seed=` argument is about 50x slower on this path (measured 2026-09-22, not used). Marked slow."""

from __future__ import annotations

import numpy as np
import pytest

from tmlm.data.synthetic import dense_config, noisy_xor, two_xor_multiclass
from tmlm.engine.oracle.oracle import OracleEngine

pytestmark = pytest.mark.slow
EPOCHS, FIRST_SCORED = 10, 4


def oracle_curve(cfg, train, test):
    eng = OracleEngine(cfg, tabular=train)
    probe = OracleEngine(cfg, tabular=test)
    accs = []
    for e in range(EPOCHS):
        for b in range(eng.n_batches()):
            eng.train_batch(e, b)
        probe.load_state(eng.state())
        pred = probe.predict_scores(0, np.arange(test.y.shape[0])).argmax(axis=1)
        accs.append(float((pred == test.y).mean()))
    assert eng.check_invariants() == []
    return accs


def tmu_curve(n_clauses, T, s, train, test):
    from tmu.models.classification.coalesced_classifier import TMCoalescedClassifier
    tm = TMCoalescedClassifier(n_clauses, T, s, platform="CPU", weighted_clauses=True)
    x_tr, y_tr, x_te = train.x.astype(np.uint32), train.y.astype(np.uint32), test.x.astype(np.uint32)
    accs = []
    for _ in range(EPOCHS):
        tm.fit(x_tr, y_tr)
        accs.append(float((tm.predict(x_te) == test.y).mean()))
    return accs


def scored(curve):
    return float(np.mean(curve[FIRST_SCORED - 1:]))


@pytest.mark.parametrize("task,n_classes", [("xor", 2), ("two_xor", 4)])
def test_dense_equivalent_mode_matches_tmu_within_two_points(task, n_classes):
    n_features, n_clauses, T, s = 12, 40, 15, 3.9
    if task == "xor":
        train, test = noisy_xor(2000, n_features, noise=0.1, seed=1), noisy_xor(1000, n_features, noise=0.0, seed=2)
    else:
        train, test = two_xor_multiclass(3000, n_features, noise=0.05, seed=1), two_xor_multiclass(1000, n_features, noise=0.0, seed=2)
    oracle = [scored(oracle_curve(dense_config(n_features, n_classes, n_clauses, T, s, seed=seed), train, test)) for seed in (1, 2, 3)]
    tmu = [scored(tmu_curve(n_clauses, T, s, train, test)) for _ in range(3)]
    print(f"{task}: oracle mean over epochs {FIRST_SCORED}..{EPOCHS}, 3 seeds: {np.mean(oracle):.4f} {oracle}; tmu: {np.mean(tmu):.4f} {tmu}")
    assert np.mean(oracle) >= np.mean(tmu) - 0.02, f"KB1: oracle {np.mean(oracle):.4f} vs tmu {np.mean(tmu):.4f}"
    assert np.mean(oracle) > 0.85
