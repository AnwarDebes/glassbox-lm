"""Tests of src/tmlm/ssl/controls.py: C0y batch count, G+ selection and feature map against a pure-Python
brute force (masked and causal literal spaces), and the count baselines (docs/13 A7)."""

from __future__ import annotations

import numpy as np
import pytest

from tmlm.data.synthetic import corpus_from_tokens, toy_corpus
from tmlm.engine.config import LIT_EMPTY, lit_unpack, offset_seg, sym_edge
from tmlm.ssl import controls as K
from tmlm.ssl import features as F


def test_c0y_batches():
    assert K.c0y_batches(20_000_000, 1024) == int(np.ceil(0.01 * np.ceil(20_000_000 / 1024)))
    assert K.c0y_batches(10, 1024) == 1
    assert K.c0y_batches(100 * 1024, 1024) == 1 and K.c0y_batches(101 * 1024, 1024) == 2


def test_columns_and_pairs():
    assert len(K.columns("masked")) == 64 and len(K.columns("causal")) == 32
    assert len(K.column_pairs("masked")) == 16 * 15 // 2 * 16
    assert len(K.column_pairs("causal")) == 8 * 7 // 2 * 16
    for a, b in K.column_pairs("masked")[:50]:
        oa, ob = K.columns("masked")[a][0], K.columns("masked")[b][0]
        assert oa < ob and offset_seg(oa) < offset_seg(ob)


def _corpus(seed=0, V=50):
    r = np.random.default_rng(seed)
    tok = r.integers(0, V, size=420).astype(np.uint16)
    ds = np.array([0, 30, 31, 100, 250, 420], dtype=np.uint32)  # includes a 1-token document
    return corpus_from_tokens(tok, ds, V, n_stop=5, kind_rng=np.random.default_rng(seed + 1))


def _ranked_bruteforce(bf: dict, n: int):
    items = sorted(bf.items(), key=lambda kv: (-kv[1], kv[0][1] != LIT_EMPTY, kv[0][0], kv[0][1]))
    return [(a, b, c) for (a, b), c in items[:n]]


@pytest.mark.parametrize("mode", ["masked", "causal", "diffusion"])
def test_gplus_select_equals_bruteforce(mode):
    V = 50
    c = _corpus(V=V)
    g = K.gplus_select(np, c.tok, c.doc_start, c.kind_map, mode, V, 400, chunk=97)
    bf = K.gplus_bruteforce(c.tok, c.doc_start, c.kind_map, mode, V)
    assert [(int(a), int(b), int(n)) for a, b, n in zip(g.lit1, g.lit2, g.count)] == _ranked_bruteforce(bf, 400)
    assert g.info["n_single"] + g.info["n_pair"] == 400 and g.info["n_pair"] > 0
    # pairs are at different offsets, literal ids ascending
    for f in np.flatnonzero(g.is_pair):
        s1, _, _ = lit_unpack(int(g.lit1[f]))
        s2, _, _ = lit_unpack(int(g.lit2[f]))
        assert s1 < s2
    # an EDGE literal exists in the literal space (short documents)
    edge = int(sym_edge(V))
    assert any(lit_unpack(int(x))[1] == edge for x in g.lit1)


@pytest.mark.parametrize("mode", ["masked", "causal"])
def test_gplus_features_equal_bruteforce_per_document(mode, tmp_path):
    V = 50
    c = _corpus(seed=3, V=V)
    g = K.gplus_select(np, c.tok, c.doc_start, c.kind_map, mode, V, 300)
    g.save(tmp_path / "g.npz")
    g2 = K.GPlus.load(tmp_path / "g.npz")
    assert np.array_equal(g2.lit1, g.lit1) and g2.mode == mode
    # a different "downstream" corpus with the same tables
    d = toy_corpus(vocab=V, n_docs=5, doc_len=23, seed=9)
    sd = K.gplus_features(np, g, d.tok, d.doc_start, c.kind_map, chunk=40)
    assert sd.n_doc == 5 and sd.n_cols == 300
    key_to_f = {(int(a), int(b)): f for f, (a, b) in enumerate(zip(g.lit1, g.lit2))}
    for doc in range(5):
        pos = range(int(d.doc_start[doc]), int(d.doc_start[doc + 1]))
        bf = K.gplus_bruteforce(d.tok, d.doc_start, c.kind_map, mode, V, positions=pos)
        exp = {key_to_f[k]: v for k, v in bf.items() if k in key_to_f}
        row = sd.rows(np.asarray([doc]))
        assert dict(zip(row.ids.tolist(), row.cnt.tolist())) == exp
    # the dimension-matched prefix of the feature map is the first columns
    sd8 = K.gplus_features(np, g, d.tok, d.doc_start, c.kind_map, n_cols=50)
    full = sd.to_csr("count", l2=False).toarray()
    assert np.array_equal(sd8.to_csr("count", l2=False).toarray(), full[:, :50])


def test_count_baselines_and_ppmi_svd():
    V = 40
    c = toy_corpus(vocab=V, n_docs=6, doc_len=120, seed=4)
    E, info = K.ppmi_svd(c.tok, c.doc_start, V, window=8, dim=10, power=0.5)
    assert E.shape == (V, 10) and np.all(np.isfinite(E)) and info["nnz_ppmi"] > 0
    E2, _ = K.ppmi_svd(c.tok, c.doc_start, V, window=8, dim=10, power=0.5)
    assert np.array_equal(E, E2)
    docs = F.TokenDocs(tok=np.array([1, 2, 3, 0, 4, 4, 0], np.uint16), doc_start=np.array([0, 4, 7], np.uint32), end_id=0)
    bow = K.bag_of_words(docs, V)
    assert bow.rows(np.array([1])).ids.tolist() == [4] and bow.rows(np.array([1])).cnt.tolist() == [2]
    cl = K.cluster_bag_of_words(docs, c.kind_map)
    assert cl.n_doc == 2
    X = K.ppmi_svd_docs(docs, E)
    assert np.allclose(X[1], E[4])
    B = K.binarise_counts(bow, np.array([0, 1]), np.array([1, 4, 5]))
    assert B.dtype == np.uint32 and B.tolist() == [[1, 0, 0], [0, 1, 0]]
    assert K.top_columns_by(np.array([5, 0, 7, 7, 1]), 2).tolist() == [2, 3]


def test_top_eigenpairs_equal_the_full_decomposition():
    from tmlm.ssl import concepts as KC

    rng = np.random.default_rng(5)
    A = rng.normal(size=(60, 40))
    G = A @ A.T
    w1, U1 = K.top_eigenpairs(G, 7)
    w2, U2 = KC.eig_desc(np, G, 7)
    assert np.allclose(w1, w2)
    assert np.allclose(np.abs(np.sum(U1 * U2, axis=0)), 1.0)  # same vectors up to sign
