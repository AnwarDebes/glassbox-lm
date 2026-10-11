"""Tests for experiments/cvtm_a3_rows.py (docs/13 amendment A3, rows H8.2, H8.7, H8.8, H8.9): the timed
full scoring pass of H8.7 (c) (suffix lookups, STREAMING document clauses, mixture) returns exactly
the P(gold) of the batch scoring path `CVTMCore.gold`, for several windows, and the suffix digest of
H8.8 (b) detects a one-bit change. Small synthetic corpora, numpy backend, no GPU."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import cvtm_a3_rows as a3  # noqa: E402
from tmlm.cvtm.doc_clauses import doc_features  # noqa: E402
from tmlm.cvtm.model import CVTMCore  # noqa: E402
from tmlm.cvtm.mixture import D_ORDERS  # noqa: E402


def corpus(seed: int, vocab: int, n_docs: int, max_len: int):
    """Documents from a skewed first-order Markov chain (rare n-grams exist, so every discount is
    positive), with an earlier span of the same document copied in (so the pointer fires)."""
    rng = np.random.default_rng(seed)
    cum = np.cumsum(rng.dirichlet(np.full(vocab, 0.2), size=vocab), axis=1)
    toks, lengths = [], []
    for _ in range(n_docs):
        n = int(rng.integers(20, max_len))
        t = [int(rng.integers(vocab))]
        for x in rng.random(n - 1):
            t.append(min(int(np.searchsorted(cum[t[-1]], x)), vocab - 1))
        t = np.asarray(t)
        if n >= 40:
            a, L = int(rng.integers(0, n // 2)), int(rng.integers(8, 16))
            b = int(rng.integers(n // 2, n - L))
            t[b:b + L] = t[a:a + L]
        toks.append(t)
        lengths.append(n)
    tok = np.concatenate(toks).astype(np.uint16)
    return tok, np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)


@pytest.fixture(scope="module")
def small_core():
    # enough documents that some have residue 0 (the fit documents H of docs/11 11.6)
    tok, ds = corpus(1, 120, 400, 160)
    core = CVTMCore.fit(tok, ds, vocab=120, backend="numpy", expect_h=False)
    ev_tok, ev_ds = corpus(2, 120, 6, 300)
    return core, ev_tok, ev_ds


@pytest.mark.parametrize("window", [None, 0, 16, 64])
def test_full_scoring_pass_equals_the_batch_path(small_core, window):
    core, tok, ds = small_core
    var = core if window is None else core.variant(window=window)
    pg, t = a3.full_scoring_pass(var, tok, ds, window)
    feats = doc_features(tok, ds, orders=D_ORDERS, window=window, ptr_min=5, ptr_max=64)
    ref = var.gold(tok, ds, feats=feats).p_gold
    assert np.array_equal(pg, ref)
    assert t["total_ns"] >= t["suffix_ns"] + t["doc_clauses_ns"]
    assert t["n_fallback"] == 0


def test_suffix_digest_changes_with_one_bit(small_core):
    core, _, _ = small_core
    sc = core.suffix
    d0 = a3.suffix_digest(sc)
    old = sc.v[3].copy()
    try:
        i = int(np.flatnonzero(sc.v[3] != 0)[0])
        sc.v[3][i] = np.nextafter(sc.v[3][i], np.inf)
        d1 = a3.suffix_digest(sc)
    finally:
        sc.v[3][:] = old
    assert d0["v[3]"] != d1["v[3]"]
    assert {k for k in d0 if d0[k] != d1[k]} == {"v[3]"}      # the int32 votes (and state hash) did not change
    assert a3.suffix_digest(sc) == d0
