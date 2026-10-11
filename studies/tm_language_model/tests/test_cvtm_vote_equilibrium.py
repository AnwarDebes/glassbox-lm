"""Tests for experiments/cvtm_vote_equilibrium.py (docs/13 amendment A4, H9.6 and gate G1 T1).

Covered: the closed form of the rest point (T (p - b)/(p + b) = T tanh(x/2), zero drift there), the
weighted R^2 of bin means, the weighted average-rank Spearman (equal to scipy on the data expanded by
integer multiplicities, ties included), the within-row Spearman kernel, the per-(group, bin) sums behind
the bootstrap, the document multiplicities, the verdict and gate rules at their boundaries, a stochastic
single-context simulation of the two gated update rules (the time average of clamp(S) settles at S*),
and one tiny planted run of the real CPU engine (code path only). All data are synthetic and seeded; no
corpus and no results/ access.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import spearmanr

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import cvtm_vote_equilibrium as ve  # noqa: E402


def test_rest_point_closed_form_and_zero_drift():
    rng = np.random.default_rng(0)
    p = np.concatenate([rng.uniform(1e-6, 0.999, 2000), [1e-4, 0.5, 0.9]])
    for V, Q, T in ((8000, 1.0, 64), (32, 1.0, 64), (256, 2.5, 16)):
        b = ve.b_of(p, Q, V)
        assert np.allclose(b, (1 - p) * Q / (V - 1), rtol=0, atol=0)
        s = ve.equilibrium(p, Q, V, T)
        assert np.allclose(s / T, np.tanh(ve.x_of(p, Q, V) / 2), rtol=0, atol=1e-12)
        # zero expected drift: p (T - S*) = b (T + S*)
        assert np.all(np.abs(p * (T - s) - b * (T + s)) <= 1e-11 * (p + b) * T)   # relative to the rates (T - S* cancels)
    # the saturation bound of the derivation: p >= 20 b  <=>  x >= ln 20  =>  S*/T >= tanh(ln(20)/2) = 19/21
    assert np.tanh(np.log(20.0) / 2) == pytest.approx(19 / 21, abs=1e-15)


def test_weighted_r2():
    n = np.array([10.0, 0.0, 5.0, 1.0])
    f = np.array([0.1, 0.9, 0.5, 0.8])
    assert ve.weighted_r2(n, f.copy(), f) == pytest.approx(1.0)
    y = np.array([0.2, 123.0, 0.4, 0.9])            # the empty bin is ignored whatever its value
    ok = n > 0
    mean = (n[ok] * y[ok]).sum() / n[ok].sum()
    want = 1 - (n[ok] * (y[ok] - f[ok]) ** 2).sum() / (n[ok] * (y[ok] - mean) ** 2).sum()
    assert ve.weighted_r2(n, y, f) == pytest.approx(want, rel=1e-12)
    # a prediction worse than the mean gives a negative R^2 (no fitted parameter)
    assert ve.weighted_r2(np.ones(3), np.array([0.0, 0.5, 1.0]), np.array([1.0, 0.5, 0.0])) < 0


def test_spearman_unit_weights_equals_scipy_with_ties():
    rng = np.random.default_rng(1)
    a = rng.integers(-5, 6, 3000)                   # many ties, like integer class scores
    b = np.round(rng.normal(size=3000) + 0.3 * a, 1)
    pa, pb = ve.RankPrep(a), ve.RankPrep(b)
    got = ve.weighted_spearman(pa, pb, np.ones(a.shape[0]))
    assert got == pytest.approx(spearmanr(a, b).statistic, abs=1e-12)


def test_spearman_multiplicities_equal_expanded_data():
    rng = np.random.default_rng(2)
    n = 500
    a = rng.integers(0, 20, n)
    b = rng.normal(size=n) + 0.1 * a
    w = rng.integers(0, 4, n).astype(np.float64)    # 0 = document not drawn, 2 or 3 = drawn repeatedly
    pa, pb = ve.RankPrep(a), ve.RankPrep(b)
    got = ve.weighted_spearman(pa, pb, w)
    idx = np.repeat(np.arange(n), w.astype(np.int64))
    assert got == pytest.approx(spearmanr(a[idx], b[idx]).statistic, abs=1e-12)


def test_within_row_spearman_kernel():
    rng = np.random.default_rng(3)
    S = rng.integers(-3, 4, (50, 12)).astype(np.int64)
    lnp = -np.sort(rng.exponential(size=(50, 12)), axis=1)
    L = rng.integers(0, 13, 50).astype(np.int64)
    S[0, :] = 7                                     # constant S: NaN
    L[0] = 12
    got = ve.k_within_row_spearman(S, lnp, L)
    for i in range(50):
        m = L[i]
        if m < 3 or np.unique(S[i, :m]).shape[0] < 2:
            assert np.isnan(got[i])
        else:
            assert got[i] == pytest.approx(spearmanr(S[i, :m], lnp[i, :m]).statistic, abs=1e-12)


def test_bin_sums_and_bootstrap_aggregation():
    rng = np.random.default_rng(4)
    x = rng.normal(size=1000) * 2
    y = np.tanh(x / 2) + rng.normal(size=1000) * 0.1
    grp = rng.integers(0, 7, 1000)
    kb = ve.bin_index(x, 0.25)
    ids, cnt, (sy,) = ve.bin_sums(kb, grp, [y], 7)
    assert cnt.sum() == 1000
    for j, k in enumerate(ids):
        m = kb == k
        assert cnt[:, j].sum() == m.sum()
        assert sy[:, j].sum() == pytest.approx(y[m].sum())
    mult = ve.doc_multiplicities(7, 50, [20260924, 1106, 1])
    assert mult.shape == (50, 7) and np.all(mult.sum(axis=1) == 7)
    assert np.array_equal(mult, ve.doc_multiplicities(7, 50, [20260924, 1106, 1]))
    # aggregated bin sums of one resample equal the sums over the expanded data
    r = 3
    idx = np.repeat(np.arange(1000), mult[r][grp].astype(np.int64))
    agg = mult[r] @ sy
    for j, k in enumerate(ids):
        assert agg[j] == pytest.approx(y[idx][kb[idx] == k].sum())


def test_verdict_boundaries():
    assert ve.verdict_h96(0.8, 0.2, 0.8, 0.2) == "MET"
    assert ve.verdict_h96(0.7999, 0.0, 0.8, 0.2) == "NOT MET"
    assert ve.verdict_h96(0.95, 0.2001, 0.8, 0.2) == "NOT MET"
    assert ve.verdict_g1t1(0.8, 0.8) == "OPEN"
    assert ve.verdict_g1t1(0.79, 0.8) == "CLOSED"


def test_stochastic_single_context_settles_at_rest_point():
    """The two gated rules of the derivation for one context and n clauses that all hold a vote
    (assumption A1), one example per step: the time average of clamp(S) settles at S* = T tanh(x/2)."""
    rng = np.random.default_rng(5)
    T, V, Q, n = 64, 64, 1.0, 8
    for p in (0.02, 0.1, 0.4):
        b = (1 - p) * Q / (V - 1)
        pi = Q / (V - 1)
        w = np.ones(n, dtype=np.int64)
        trace = []
        for t in range(60000):
            s = int(w.sum())
            cs = min(T, max(-T, s))
            if rng.random() < p:                    # target: every clause gated with (T - clamp S)/(2T)
                w += rng.random(n) < (T - cs) / (2 * T)
            else:                                   # negative: drawn w.p. Q/(V-1), gated with (T + clamp S)/(2T)
                w -= (rng.random(n) < pi) & (rng.random(n) < (T + cs) / (2 * T))
            if t >= 10000:
                trace.append(cs)
        want = ve.equilibrium(np.array([p]), Q, V, T)[0]
        assert np.mean(trace) == pytest.approx(want, abs=0.06 * T), (p, np.mean(trace), want)
        assert abs(p - b) > 0


def test_planted_code_path_runs():
    pc = {"n_features": 4, "n_train": 600, "n_test": 50, "epochs": 1, "n_clauses": 64, "T": 16, "Q": 1.0,
          "zipf_a": 1.0, "snapshots_last_epoch": 2}
    r = ve.planted_one(pc, V=8, B=32, seed=100)
    assert len(r["mean_S_over_T"]) == 8 and r["snapshots"] == 2
    assert np.isfinite(r["r2_clamped"]) or np.isnan(r["r2_clamped"])
