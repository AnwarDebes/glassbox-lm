"""Fixed combination of the CVTM (docs/11 section 11.4): bucket layout, simplex, determinism, EM
monotonicity, gold consistency with the full distribution, normalisation, the vote form, the H8.4
component restriction, and recovery of planted mixture weights."""

from __future__ import annotations

import numpy as np
import pytest

from tmlm.cvtm.doc_clauses import doc_candidates, doc_features
from tmlm.cvtm.mixture import (CACHE_COMPONENTS, COMPONENTS, N_COMP, BucketLayout, active_mask, bucket_ids,
                               count_bucket, counts_matrix, fit_map_em, full_distribution, gold_components,
                               gold_prob, pointer_bucket, pooled_em, residual_votes)


# ------------------------------------------------------------------ synthetic data
def synthetic(seed=0, n=6000, vocab=30, true_w=None):
    """Rows with bucket-determined active sets; the gold is drawn from the planted mixture."""
    rng = np.random.default_rng(seed)
    lay = BucketLayout.of(None)
    k = rng.integers(0, 5, size=n)
    m_star = rng.integers(-1, 5, size=n)
    c = rng.integers(0, 5, size=n)
    l = np.where(m_star == 4, rng.integers(0, 4, size=n), 0)
    bucket = lay.encode(k, m_star, c, l)
    act = np.stack([lay.active_components(int(b)) for b in bucket])
    dists = rng.dirichlet(np.full(vocab, 0.3), size=(n, N_COMP))
    dists[:, 6] = 0.0
    ptr = rng.integers(0, vocab, size=n)
    dists[np.arange(n), 6, ptr] = 1.0
    if true_w is None:
        true_w = rng.dirichlet(np.ones(N_COMP), size=lay.n_buckets)
    probs = np.where(act, true_w[bucket], 0.0)
    probs /= probs.sum(axis=1, keepdims=True)
    z = np.array([rng.choice(N_COMP, p=pr) for pr in probs])
    gold = np.array([rng.choice(vocab, p=dists[i, z[i]]) for i in range(n)])
    p = dists[np.arange(n), :, gold]
    return p, act, bucket, lay, dists, ptr, gold, true_w


# ------------------------------------------------------------------ layout
def test_bucket_layout_counts_and_roundtrip():
    full = BucketLayout.of(None)
    cache = BucketLayout.of(CACHE_COMPONENTS)
    assert full.n_buckets == 5 * 26 * 4 == 520
    assert cache.n_buckets == 5 * 6 * 1 == 30
    for lay in (full, cache):
        seen = set()
        for b in range(lay.n_buckets):
            k, m, c, l = lay.decode(b)
            again = int(lay.encode(k, -1 if m is None else m, 0 if c is None else c, l))
            assert again == b
            seen.add(b)
        assert len(seen) == lay.n_buckets
    assert count_bucket([0, 1, 2, 3, 4, 7, 8, 15, 16, 999]).tolist() == [-1, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    assert pointer_bucket([0, 4, 5, 8, 9, 16, 17, 64]).tolist() == [0, 0, 1, 1, 2, 2, 3, 3]


def test_bucket_ids_and_active_mask_from_doc_features():
    rng = np.random.default_rng(1)
    tok = rng.integers(0, 4, size=900)
    ds = np.array([0, 400, 900])
    f = doc_features(tok, ds)
    k_star = rng.integers(0, 5, size=900)
    nc = counts_matrix(f)
    for comps in (None, CACHE_COMPONENTS):
        lay = BucketLayout.of(comps)
        b, nb = bucket_ids(k_star, nc, f.ptr_len, comps)
        assert nb == lay.n_buckets and b.min() >= 0 and b.max() < nb
        act = active_mask(nc, f.ptr_len, comps)
        implied = np.stack([lay.active_components(int(x)) for x in b])
        np.testing.assert_array_equal(act, implied)       # the bucket fixes the active set
        p, act2 = gold_components(np.full(900, 0.01), f, tok)
        if comps is None:
            np.testing.assert_array_equal(act2, act)


# ------------------------------------------------------------------ fitting
def test_simplex_determinism_and_monotonicity():
    p, act, bucket, lay, *_ = synthetic(0, n=4000)
    tr1, tr2 = {}, {}
    w1 = fit_map_em(p, act, bucket, n_buckets=lay.n_buckets, trace=tr1)
    w2 = fit_map_em(p, act, bucket, n_buckets=lay.n_buckets, trace=tr2)
    assert w1.shape == (520, 7)
    assert w1.tobytes() == w2.tobytes()                   # identical bytes
    assert np.all(w1 >= 0.0) and np.allclose(w1.sum(axis=1), 1.0, atol=1e-12, rtol=0)
    ll = np.asarray(tr1["pooled_ll"])
    assert ll.shape == (51,) and np.all(np.diff(ll) >= -1e-12)
    obj = np.asarray(tr1["bucket_map_objective"])
    assert np.all(np.diff(obj) >= -1e-12)
    # zero weight outside the bucket's active set
    for b, names in tr1["active_sets"].items():
        outside = [i for i, c in enumerate(COMPONENTS) if c not in names]
        assert np.all(w1[b, outside] == 0.0)


def test_pooled_em_monotone_with_heterogeneous_active_sets():
    rng = np.random.default_rng(5)
    n = 3000
    act = rng.random((n, N_COMP)) < 0.5
    act[:, 0] = True
    p = rng.random((n, N_COMP)) * act
    tr = {}
    w = pooled_em(p, act, iters=80, trace=tr)
    assert np.all(np.diff(tr["pooled_ll"]) >= -1e-12)
    assert abs(w.sum() - 1.0) < 1e-12 and np.all(w >= 0)


def test_recovers_planted_weights():
    lay = BucketLayout.of(None)
    rng = np.random.default_rng(7)
    true_w = rng.dirichlet(np.full(N_COMP, 2.0), size=lay.n_buckets)
    # few buckets with many rows each
    n = 60000
    k = np.zeros(n, np.int64)
    m_star = np.where(np.arange(n) % 2 == 0, 4, 1)
    c = np.full(n, 2)
    l = np.where(m_star == 4, 3, 0)
    bucket = lay.encode(k, m_star, c, l)
    act = np.stack([lay.active_components(int(b)) for b in np.unique(bucket)])
    act = act[np.searchsorted(np.unique(bucket), bucket)]
    vocab = 50
    dists = rng.dirichlet(np.full(vocab, 0.2), size=(n, N_COMP))
    probs = np.where(act, true_w[bucket], 0.0)
    probs /= probs.sum(axis=1, keepdims=True)
    u = rng.random(n)
    z = (u[:, None] > np.cumsum(probs, axis=1)).sum(axis=1)
    cum = np.cumsum(dists[np.arange(n), z], axis=1)
    gold = np.minimum((rng.random(n)[:, None] > cum).sum(axis=1), vocab - 1)
    p = dists[np.arange(n), :, gold]
    w = fit_map_em(p, act, bucket, iters=400, prior=0.0, n_buckets=lay.n_buckets)
    for b in np.unique(bucket):
        a = lay.active_components(int(b))
        want = true_w[b] * a
        want = want / want.sum()
        assert np.max(np.abs(w[b] - want)) < 0.03, (b, w[b], want)


def test_prior_pulls_small_buckets_to_pooled():
    p, act, bucket, lay, *_ = synthetic(3, n=3000)
    tr = {}
    w = fit_map_em(p, act, bucket, prior=1e9, n_buckets=lay.n_buckets, trace=tr)
    pooled = np.asarray(tr["pooled_weights"])
    for b, names in tr["active_sets"].items():
        a = np.array([c in names for c in COMPONENTS])
        want = pooled * a / (pooled * a).sum()
        assert np.max(np.abs(w[b] - want)) < 1e-5
    empty = [b for b in range(lay.n_buckets) if tr["rows_per_bucket"][b] == 0]
    assert empty and np.allclose(w[empty], pooled[None, :])


def test_mixed_active_sets_in_one_bucket_are_refused():
    p, act, bucket, lay, *_ = synthetic(4, n=500)
    act2 = act.copy()
    act2[0, 1:] = ~act2[0, 1:]
    act2[0, 0] = True
    with pytest.raises(ValueError):
        fit_map_em(p, act2, np.zeros_like(bucket), n_buckets=1)


# ------------------------------------------------------------------ scoring
def test_gold_consistency_normalisation_and_votes():
    rng = np.random.default_rng(11)
    vocab = 40
    tok = rng.integers(0, 6, size=1500)
    for i in range(0, 1400, 97):                         # copied spans: pointer fires
        tok[i + 50:i + 62] = tok[i:i + 12]
    ds = np.array([0, 700, 1500])
    f = doc_features(tok, ds)
    base = rng.dirichlet(np.ones(vocab), size=tok.shape[0])
    p_kn = base[np.arange(tok.shape[0]), tok]
    k_star = rng.integers(0, 5, size=tok.shape[0])
    p, act = gold_components(p_kn, f, tok)
    assert act[:, 6].any() and act[:, 5].any()
    for comps in (None, CACHE_COMPONENTS):
        b, nb = bucket_ids(k_star, counts_matrix(f), f.ptr_len, comps)
        w = fit_map_em(p, act, b, n_buckets=nb, components=comps)
        pg = gold_prob(p, act, b, w, comps)
        pos = np.arange(0, tok.shape[0], 7)
        cands = doc_candidates(tok, ds, pos)
        for r, i in enumerate(pos):
            rows = [cands[m].row(r) for m in range(5)]
            ptr = int(f.ptr_cand[i]) if f.ptr_len[i] >= 5 else -1
            full = full_distribution(base[i], rows, ptr, w[b[i]], comps)
            assert abs(full.sum() - 1.0) < 1e-12
            assert full[tok[i]] == pg[i]                   # bit for bit
            dv, ct, cv = residual_votes(base[i], rows, ptr, w[b[i]], comps)
            rv = np.full(vocab, dv)
            rv[ct] = cv
            np.testing.assert_allclose(base[i] * np.exp(rv), full, rtol=1e-12, atol=1e-15)


def test_cache_components_ignore_other_columns():
    p, act, bucket, lay, *_ = synthetic(9, n=2000)
    act_c = act.copy()
    act_c[:, 2:] = False
    lay_c = BucketLayout.of(CACHE_COMPONENTS)
    k, m, c, l = zip(*[lay.decode(int(x)) for x in bucket])
    m = np.array([-1 if x is None else x for x in m])
    c = np.array([0 if x is None else x for x in c])
    m_c = np.where(m >= 0, 0, -1)
    b_c = lay_c.encode(np.array(k), m_c, np.where(m >= 0, c, 0), np.zeros(len(k), np.int64))
    # D0 firing whenever any D fires (nesting), with the count bucket of the top order reused
    w1 = fit_map_em(p, act, b_c, n_buckets=lay_c.n_buckets, components=CACHE_COMPONENTS)
    w2 = fit_map_em(np.where(act_c, p, np.nan), act_c, b_c, n_buckets=lay_c.n_buckets, components=CACHE_COMPONENTS)
    assert w1.tobytes() == w2.tobytes()
    assert np.all(w1[:, 2:] == 0.0)


def test_kn_gold_and_depth_matches_knmodel():
    from tmlm.cvtm.mixture import kn_gold_and_depth
    from tmlm.eval.baselines.ngram import KNModel

    rng = np.random.default_rng(21)
    vocab = 12
    tr = rng.integers(0, vocab, size=3000)
    tr_ds = np.array([0, 700, 1900, 3000])
    ev = rng.integers(0, vocab, size=400)
    ev[200:230] = tr[100:130]                            # long contexts that occur in training
    ev_ds = np.array([0, 150, 400])
    kn = KNModel(5, vocab).fit(tr, tr_ds)
    p, k_star = kn_gold_and_depth(kn, ev, ev_ds, chunk=97)
    np.testing.assert_array_equal(np.log2(p), kn.logprob(ev, ev_ds))
    bos = vocab

    def left(tok, ds, j, k):
        a = ds[np.searchsorted(ds, j, side="right") - 1]
        return tuple(int(tok[j - i]) if j - i >= a else bos for i in range(k, 0, -1))

    seen = {k: {left(tr, tr_ds, j, k) for j in range(tr.shape[0])} for k in range(1, 5)}
    for j in range(ev.shape[0]):
        want = max([0] + [k for k in range(1, 5) if left(ev, ev_ds, j, k) in seen[k]])
        assert k_star[j] == want, j
    assert k_star[0] == 4 and k_star[150] == 4            # all-BOS contexts occur at every document start
