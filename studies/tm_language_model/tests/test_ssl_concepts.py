"""Tests of src/tmlm/ssl/concepts.py (the early concept test, exploratory, docs/12 F23)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tmlm.ssl import concepts as K


def _dense_counts(rng, V=7, F=11, density=0.5):
    X = rng.integers(1, 20, size=(V, F)).astype(np.float64)
    X[rng.random((V, F)) > density] = 0
    X[0, :] = 0  # an empty row
    return X


def _coo(X):
    r, c = np.nonzero(X)
    return r, c, X[r, c]


def test_ppmi_matches_brute_force():
    rng = np.random.default_rng(1)
    X = _dense_counts(rng)
    r, c, v = _coo(X)
    p = K.add_one_unigram(rng.integers(0, 7, 50), 7)
    m = K.marginals_from_coo(r, c, v, 7, 11, p)
    w = K.weight_values(np, r, c, v, m.to(np), "ppmi")
    R, N = X.sum(1), X.sum(0)
    Z = (N[N > 0] ** 0.75).sum()
    for (i, j), x in zip(zip(r, c), w):
        ref = max(0.0, math.log(X[i, j] * Z / (R[i] * N[j] ** 0.75)))
        assert abs(float(x) - ref) < 1e-5


def test_vote_is_the_witten_bell_vote_of_t1():
    rng = np.random.default_rng(2)
    X = _dense_counts(rng)
    r, c, v = _coo(X)
    p = K.add_one_unigram(rng.integers(0, 7, 80), 7)
    m = K.marginals_from_coo(r, c, v, 7, 11, p)
    w = K.weight_values(np, r, c, v, m.to(np), "vote")
    N = X.sum(0)
    D = (X > 0).sum(0)
    for (i, j), x in zip(zip(r, c), w):
        lam = N[j] / (N[j] + D[j])
        q = math.log(1 + lam * X[i, j] / (N[j] * (1 - lam) * p[i]))  # docs/11 11.5, experiments/cvtm_t1_readout.py
        assert abs(float(x) - q) < 1e-5


def test_raw_weight_and_unknown_scheme():
    r, c, v = np.array([0, 1]), np.array([0, 0]), np.array([3.0, 4.0])
    m = K.marginals_from_coo(r, c, v, 2, 1, np.array([0.5, 0.5]))
    assert np.array_equal(K.weight_values(np, r, c, v, m.to(np), "raw"), np.array([3.0, 4.0], np.float32))
    with pytest.raises(ValueError):
        K.weight_values(np, r, c, v, m.to(np), "bogus")


def test_blockwise_gram_equals_dense_and_sparse():
    rng = np.random.default_rng(3)
    X = _dense_counts(rng, V=6, F=20)
    G = np.zeros((6, 6))
    for c0 in range(0, 20, 7):
        blk = X[:, c0:c0 + 7]
        r, c, v = _coo(blk)
        K.gram_add_dense(np, G, r, c, v.astype(np.float32), 6, blk.shape[1])
    assert np.allclose(G, X @ X.T)
    r, c, v = _coo(X)
    assert np.allclose(K.gram_sparse(r, c, v, 6, 20), X @ X.T)


def test_cosine_and_spectral_embedding():
    rng = np.random.default_rng(4)
    X = _dense_counts(rng, V=6, F=9, density=0.7)
    G = X @ X.T
    S = K.cosine_from_gram(np, G)
    n = np.linalg.norm(X, axis=1)
    for a in range(6):
        for b in range(6):
            ref = 0.0 if n[a] == 0 or n[b] == 0 else X[a] @ X[b] / (n[a] * n[b])
            assert abs(S[a, b] - ref) < 1e-5
    w, U = K.eig_desc(np, G, 6)
    assert np.all(np.diff(w) <= 1e-9)
    E = K.spectral_embedding(np, w, U, 6, 1.0)  # full rank, p = 1: E E^T = G
    assert np.allclose(E @ E.T, G, atol=1e-3 * G.max())
    assert np.allclose(K.cosine_rows(np, E), S, atol=1e-4)
    assert K.parse_reduction("full") is None and K.parse_reduction("svd300_p0.5") == (300, 0.5)


def test_cooc_counts_brute_force():
    rng = np.random.default_rng(5)
    tok = rng.integers(0, 5, 40)
    ds = np.array([0, 13, 13, 29, 40])  # one empty document
    mask = rng.random(40) < 0.7
    offs = [-2, -1, 1, 3]
    X = K.cooc_counts(tok, ds, 5, offs, target_mask=mask, chunk=7)
    ref = np.zeros((5, 5), np.int64)
    doc = np.repeat(np.arange(4), np.diff(ds))
    for i in range(40):
        if not mask[i]:
            continue
        for o in offs:
            j = i + o
            if 0 <= j < 40 and doc[j] == doc[i]:
                ref[tok[i], tok[j]] += 1
    assert np.array_equal(X, ref)


def test_literal_profile_tiny_state():
    from tmlm.engine.config import lit_pack
    V = 10
    lit_id = np.full((3, 4), 0xFFFFFFFF, np.uint32)
    lit_state = np.zeros((3, 4), np.uint8)
    # clause 0: t[-1]=3, NOT t[-2]=5, a cluster literal (sym >= V, dropped), an excluded literal (state < 128)
    lit_id[0] = [lit_pack(7, 3, 0), lit_pack(6, 5, 1), lit_pack(7, V + 2, 0), lit_pack(5, 4, 0)]
    lit_state[0] = [200, 150, 180, 100]
    # clause 1: t[-1]=3 and a bag literal for 4; clause 2 dead
    lit_id[1, :2] = [lit_pack(7, 3, 0), lit_pack(16 + 2 * 1, 4, 0)]
    lit_state[1, :2] = [130, 140]
    lit_id[2, 0], lit_state[2, 0] = lit_pack(7, 1, 0), 200
    alive = np.array([1, 1, 0], np.uint8)
    r, c, n = K.literal_profile(lit_id, lit_state, alive, V, offset_bound=False)
    assert n == 6
    assert sorted(zip(r.tolist(), c.tolist())) == [(3, 0), (3, 2), (4, 2), (5, 1)]
    r2, c2, n2 = K.literal_profile(lit_id, lit_state, alive, V, offset_bound=True)
    assert n2 == 2 * 3 * K.N_SEG_ALL
    assert sorted(zip(r2.tolist(), c2.tolist())) == sorted([(3, 2 * (0 * 42 + 7)), (5, 2 * (0 * 42 + 6) + 1), (3, 2 * (1 * 42 + 7)), (4, 2 * (1 * 42 + 18))])


def test_spearman_and_bootstrap():
    from scipy.stats import spearmanr
    rng = np.random.default_rng(6)
    g = rng.normal(size=50)
    p = g + rng.normal(size=50)
    p[:5] = 0.0  # ties
    assert abs(K.spearman(g, p) - spearmanr(g, p).statistic) < 1e-12
    idx = K.bootstrap_indices(50, 30, 100)
    assert np.array_equal(idx, K.bootstrap_indices(50, 30, 100))
    bs = K.spearman_boot(g, p, idx)
    for k in (0, 7, 29):
        assert abs(bs[k] - K.spearman(g[idx[k]], p[idx[k]])) < 1e-12
    lo, hi = K.ci95(bs)
    assert lo <= hi


def test_parsers_and_pair_mapping(tmp_path):
    s = tmp_path / "s.txt"
    s.write_text("word1\tword2\tPOS\tSimLex999\tconc(w1)\n old\tnew\tA\t1.58\t2.7\n".replace("\n old", "\nold"))
    assert K.parse_simlex(s) == [("old", "new", 1.58)]
    w = tmp_path / "w.csv"
    w.write_text("Word 1,Word 2,Human (mean)\nlove,sex,6.77\nFBI,fingerprint,6.94\n")
    assert K.parse_ws353(w)[1] == ("FBI", "fingerprint", 6.94)
    m = tmp_path / "m.dev"
    m.write_text("berry-n seed-n 37.000000\n")
    assert K.parse_men_lemma(m) == [("berry", "seed", 37.0)]
    vocab = {" old": 5, " new": 6, " love": 7, " fbi": 8}
    enc = lambda x: [vocab[x]] if x in vocab else [1, 2]  # noqa: E731
    assert K.single_token_id(enc, "FBI") == 8 and K.single_token_id(enc, "FBI", try_lower=False) is None
    mp = K.map_pairs([("old", "new", 1.0), ("love", "sex", 2.0), ("FBI", "old", 3.0)], enc)
    assert mp["n_covered"] == 2 and mp["a"].tolist() == [5, 8] and mp["b"].tolist() == [6, 5] and mp["gold"].tolist() == [1.0, 3.0]


def test_word_like_mask():
    assert K.word_like_mask([" dog", "dog", " Dog", " an", " café", " dogs"]).tolist() == [True, False, False, False, False, True]


def test_neighbour_metrics_and_purity():
    neigh = np.array([[1, 2], [0, 3]])
    lab = {0: "a", 1: "a", 2: "b", 3: "b"}
    assert K.neighbour_agreement(neigh, ["a", "a"], lab) == pytest.approx(0.5)  # rows: (a, b) and (a, b)
    assert K.neighbour_agreement(np.array([[1, 0]]), ["a"], lab) == pytest.approx(1.0)
    syn = {0: {"x"}, 1: {"x"}, 2: {"y"}, 3: {"z"}}
    assert K.synonym_hit_rate(neigh, np.array([0, 1]), syn) == 1.0
    assert K.label_chance(["a", "a", "b", "b"]) == pytest.approx(0.5)
    assert K.purity(np.array([0, 0, 1, 1]), ["a", "a", "b", "a"]) == pytest.approx(0.75)
    rng = np.random.default_rng(7)
    E = np.concatenate([rng.normal(size=(20, 3)) + [8, 0, 0], rng.normal(size=(20, 3)) + [0, 8, 0]])
    a1 = K.kmeans_cosine(E, 2, 100)
    assert np.array_equal(a1, K.kmeans_cosine(E, 2, 100))
    assert K.purity(a1, ["p"] * 20 + ["q"] * 20) == 1.0
    assert K.permuted_purity(a1, ["p"] * 20 + ["q"] * 20, 5, 100) < 1.0


def test_clause_rule_text():
    from tmlm.engine.config import lit_pack
    V = 10
    ids = np.array([lit_pack(7, 3, 0), lit_pack(6, 5, 1), lit_pack(7, V + 2, 0), 0xFFFFFFFF], np.uint32)
    st = np.array([200, 150, 180, 0], np.uint8)
    txt = K.clause_rule(ids, st, V, lambda t: f" w{t}", lambda sym: [" a", " b"])
    assert txt.startswith("IF ") and "t[-1]=' w3'" in txt and "NOT t[-2]=' w5'" in txt and "c4:2 {' a',' b'}" in txt
