"""Tests of src/tmlm/ssl/probes.py: stratified nested label draws, the 4-fold LR probe, the Coalesced TM
probe (deterministic), Q_text and the paired bootstrap (docs/13 A7)."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from tmlm.ssl import probes as PR


def _labels(n=400, k=4, seed=0):
    rng = np.random.default_rng(seed)
    return rng.choice(k, size=n, p=[0.5, 0.3, 0.15, 0.05][:k] if k == 4 else None)


def test_stratified_draws_are_nested_and_deterministic():
    y = _labels()
    counts = np.bincount(y)
    prev = None
    for n in (8, 32, 128):
        s = PR.label_sample(y, n, task_code=2, draw=0)
        assert np.all(np.diff(s) > 0)
        got = np.bincount(y[s], minlength=4)
        assert np.all(got == np.minimum(n, counts))
        if prev is not None:
            assert set(prev.tolist()) <= set(s.tolist())
        prev = s
    assert np.array_equal(PR.label_sample(y, 32, 2, 1), PR.label_sample(y, 32, 2, 1))
    assert not np.array_equal(PR.label_sample(y, 32, 2, 0), PR.label_sample(y, 32, 2, 1))
    assert not np.array_equal(PR.label_sample(y, 32, 2, 0), PR.label_sample(y, 32, 3, 0))
    assert np.array_equal(PR.label_sample(y, "full", 2, 0), np.arange(y.shape[0]))
    u = PR.sample_union(y, [8, 32], [0, 1], 2)
    exp = np.unique(np.concatenate([PR.label_sample(y, b, 2, d) for b in (8, 32) for d in (0, 1)]))
    assert np.array_equal(u, exp)


def _separable(n_per=40, k=3, dim=60, seed=0):
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(k), n_per)
    X = (rng.random((y.shape[0], dim)) < 0.05).astype(np.float64)
    for c in range(k):
        X[y == c, c * 5:(c + 1) * 5] += (rng.random(((y == c).sum(), 5)) < 0.8)
    return X, y


def test_lr_probe_uses_4_folds_inside_the_labelled_sample():
    X, y = _separable()
    lab = PR.label_sample(y, 8, task_code=1, draw=0)
    ev = np.setdiff1d(np.arange(y.shape[0]), lab)
    c = PR.lr_probe(sp.csr_matrix(X[lab]), y[lab], sp.csr_matrix(X[ev]), y[ev], seed=5)
    assert c.cv["folds"] == 4 and c.n_labelled == 24 and c.n_eval == ev.shape[0]
    assert c.choice in PR.C_GRID
    assert c.accuracy > 0.8
    c2 = PR.lr_probe(sp.csr_matrix(X[lab]), y[lab], sp.csr_matrix(X[ev]), y[ev], seed=5)
    assert c2.accuracy == c.accuracy and c2.choice == c.choice


def test_tm_probe_is_deterministic_and_learns():
    X, y = _separable(n_per=32, dim=40)
    Xb = (X > 0).astype(np.uint32)
    lab = PR.label_sample(y, 16, task_code=1, draw=0)
    ev = np.setdiff1d(np.arange(y.shape[0]), lab)
    grid = ({"clauses": 40, "T": 20, "s": 3.0},)
    a = PR.tm_probe(Xb[lab], y[lab], Xb[ev], y[ev], seed=3, grid=grid, epochs=5)
    b = PR.tm_probe(Xb[lab], y[lab], Xb[ev], y[ev], seed=3, grid=grid, epochs=5)
    assert a.accuracy == b.accuracy and a.cv == b.cv
    assert a.cv["folds"] == 4
    assert a.accuracy > 0.6


def test_q_text_needs_all_three_tasks():
    assert PR.q_text({"agnews": 0.8, "sst2": 0.6, "trec": 0.4}) == pytest.approx(0.6)
    with pytest.raises(KeyError):
        PR.q_text({"agnews": 0.8, "sst2": 0.6})


def test_paired_bootstrap():
    r = PR.paired_bootstrap(np.array([0.02, 0.03, 0.025, 0.01, 0.04]), n_boot=2000)
    assert r["lo"] > 0 and r["excludes_0"] and r["mean"] == pytest.approx(0.025)
    r0 = PR.paired_bootstrap(np.array([0.02, -0.03, 0.01, -0.01, 0.0]), n_boot=2000)
    assert not r0["excludes_0"]
    assert PR.paired_bootstrap(np.array([0.1, 0.2]), n_boot=500) == PR.paired_bootstrap(np.array([0.1, 0.2]), n_boot=500)
