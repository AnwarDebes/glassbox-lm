"""Tests of src/tmlm/ssl/features.py and src/tmlm/ssl/transfer.py (the early transfer test)."""

from __future__ import annotations

import numpy as np
import pytest

from tmlm.ssl import features as F
from tmlm.ssl import transfer as TR


def test_wikitext_style_matches_crtm():
    crtm = pytest.importorskip("tmlm.cvtm.crtm")
    for s in ["It isn't a charming film.", "Paul's dog, however; ran!", "the boys' toys", "  a  b  ", "Wall St. Bears Claw Back (Reuters)",
              "what 's up?", "I'm sure they'll say we'd've"]:
        assert F.wikitext_style(s) == crtm.wikitext_style(s)


def test_doc_text_ends_with_end_marker():
    assert F.doc_text("A film.") == " A film . \n"


class _FakeEnc:
    def __init__(self, ids):
        self.ids = ids


class _FakeTok:
    """Character-level fake tokenizer: ' \\n' is one token (id 1), every other character its ord() + 2."""

    def get_vocab_size(self):
        return 300

    def encode(self, text):
        return _FakeEnc(self._ids(text))

    def encode_batch(self, texts):
        return [self.encode(t) for t in texts]

    @staticmethod
    def _ids(text):
        out, i = [], 0
        while i < len(text):
            if text[i:i + 2] == " \n":
                out.append(1)
                i += 2
            else:
                out.append(ord(text[i]) % 250 + 2)
                i += 1
        return out


def test_encode_and_masks():
    docs = F.encode(_FakeTok(), ["ab", "c"])
    assert docs.n_doc == 2
    assert docs.end_id == 1
    ds = docs.doc_start.astype(int)
    assert all(docs.tok[ds[1:] - 1] == 1)
    tm = docs.text_mask()
    assert tm.sum() == docs.n_tok - 2
    assert list(docs.doc_of_pos()) == [0] * (ds[1] - ds[0]) + [1] * (ds[2] - ds[1])


def test_concat_offsets():
    a = F.TokenDocs(np.array([5, 1, 6, 7, 1], np.uint16), np.array([0, 2, 5], np.uint32), 1)
    b = F.TokenDocs(np.array([8, 1], np.uint16), np.array([0, 2], np.uint32), 1)
    c, off = F.concat([a, b])
    assert list(off) == [0, 2, 3]
    assert list(c.doc_start) == [0, 2, 5, 7]
    assert list(c.tok) == [5, 1, 6, 7, 1, 8, 1]


def test_aggregate_and_rows_and_df():
    d = np.array([0, 0, 0, 2, 2, 1])
    j = np.array([5, 3, 5, 7, 7, 3])
    sd = F.aggregate_doc_counts(d, j, 3, 10)
    assert list(sd.indptr) == [0, 2, 3, 4]
    assert list(sd.ids) == [3, 5, 3, 7]
    assert list(sd.cnt) == [1, 2, 1, 2]
    sub = sd.rows(np.array([2, 0]))
    assert list(sub.ids) == [7, 3, 5] and list(sub.cnt) == [2, 1, 2]
    df = sd.doc_freq()
    assert df[3] == 2 and df[5] == 1 and df[7] == 1
    assert np.array_equal(sd.doc_freq_range(0, 3), df)
    assert np.array_equal(sd.doc_freq_range(1, 3), sd.doc_freq(np.array([1, 2])))
    merged = F.merge_rows([sd.rows(np.array([0])), sd.rows(np.array([1, 2]))])
    assert np.array_equal(merged.indptr, sd.indptr) and np.array_equal(merged.ids, sd.ids)


def test_to_csr_variants():
    sd = F.aggregate_doc_counts(np.array([0, 0, 0, 1]), np.array([2, 2, 4, 0]), 3, 5)
    X = sd.to_csr("count", l2=False).toarray()
    assert X[0, 2] == 2 and X[0, 4] == 1 and X[1, 0] == 1 and X[2].sum() == 0
    Xb = sd.to_csr("binary", l2=False).toarray()
    assert Xb[0, 2] == 1
    Xl = sd.to_csr("log1p", l2=True).toarray()
    assert np.allclose(np.linalg.norm(Xl[:2], axis=1), 1.0) and Xl[2].sum() == 0
    Xc = sd.to_csr("binary", cols=np.array([4, 0]), l2=False).toarray()
    assert Xc.shape == (3, 2) and Xc[0, 0] == 1 and Xc[1, 1] == 1 and Xc[0, 1] == 0


def test_top_df_columns_ties_and_zero():
    df = np.array([3, 5, 5, 0, 1])
    assert list(F.top_df_columns(df, 2)) == [1, 2]
    assert list(F.top_df_columns(df, 3)) == [0, 1, 2]
    assert list(F.top_df_columns(df, 10)) == [0, 1, 2, 4]


def test_bag_of_tokens_excludes_end_and_crosses_no_boundary():
    docs = F.TokenDocs(np.array([4, 5, 4, 1, 6, 1], np.uint16), np.array([0, 4, 6], np.uint32), 1)
    uni = F.bag_of_tokens(docs, 10)
    assert list(uni.ids[:uni.indptr[1]]) == [4, 5] and list(uni.cnt[:uni.indptr[1]]) == [2, 1]
    assert list(uni.ids[uni.indptr[1]:]) == [6]
    bi = F.bag_of_tokens(docs, 10, bigrams=True)
    row0 = set(bi.ids[:bi.indptr[1]].tolist())
    assert row0 == {4, 5, 10 + 4 * 10 + 5, 10 + 5 * 10 + 4}
    row1 = set(bi.ids[bi.indptr[1]:].tolist())
    assert row1 == {6}            # no bigram with END, none across documents
    comp, used = F.compact_columns(bi)
    assert comp.n_cols == used.shape[0] and np.array_equal(used[comp.ids.astype(int)], bi.ids.astype(int))


def test_cluster_bags_and_mean_embedding():
    km = np.stack([np.arange(6), np.array([10, 10, 11, 11, 12, 12]), np.array([20, 21, 22, 23, 24, 25]),
                   np.array([30, 30, 30, 31, 31, 31])]).astype(np.uint16)
    docs = F.TokenDocs(np.array([0, 2, 1, 3, 1], np.uint16), np.array([0, 3, 5], np.uint32), 1)
    cb, sym = F.cluster_bags(docs, km)
    dense = np.zeros((2, cb.n_cols))
    for d in range(2):
        for k in range(cb.indptr[d], cb.indptr[d + 1]):
            dense[d, cb.ids[k]] = cb.cnt[k]
    col = {int(s): i for i, s in enumerate(sym)}
    assert dense[0, col[10]] == 1 and dense[0, col[11]] == 1 and dense[0, col[30]] == 2
    assert dense[1, col[11]] == 1 and dense[1, col[31]] == 1 and dense[1].sum() == 3
    W = np.arange(12, dtype=float).reshape(6, 2)
    E = F.mean_embedding(docs, W)
    assert np.allclose(E[0], (W[0] + W[2]) / 2) and np.allclose(E[1], W[3])


def test_thermometer():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    edges = F.quantile_edges(X, 3)
    B = F.thermometer(X, edges)
    assert B.shape == (4, 3) and B.dtype == np.uint32
    assert list(B.sum(axis=1)) == sorted(B.sum(axis=1).tolist())


def test_label_sample_deterministic_and_full():
    a = TR.label_sample(1000, 64, 1, 101)
    b = TR.label_sample(1000, 64, 1, 101)
    c = TR.label_sample(1000, 64, 1, 102)
    assert np.array_equal(a, b) and not np.array_equal(a, c) and a.shape == (64,) and np.all(np.diff(a) > 0)
    assert np.array_equal(TR.label_sample(50, "full", 1, 101), np.arange(50))
    assert np.array_equal(TR.label_sample(50, 64, 1, 101), np.arange(50))


def test_t_interval():
    ci = TR.t_interval([0.5, 0.6, 0.7])
    assert abs(ci["mean"] - 0.6) < 1e-12
    assert abs((ci["hi"] - ci["mean"]) - 4.303 * 0.1 / np.sqrt(3)) < 1e-9
    assert TR.t_interval([0.5])["lo"] is None


def test_lr_cell_separable():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    y = (X[:, 0] > 0).astype(int)
    r = TR.lr_cell(X[:100], y[:100], X[100:], y[100:], seed=1, dense=True)
    assert r.accuracy > 0.9 and r.choice in TR.C_GRID


def test_tm_cell_learns_a_literal():
    pytest.importorskip("tmu")
    rng = np.random.default_rng(0)
    X = (rng.random((300, 16)) < 0.5).astype(np.uint32)
    y = X[:, 3].astype(np.int64)
    grid = [{"clauses": 20, "T": 10, "s": 3.0, "epochs": 5}]
    r = TR.tm_cell(X[:200], y[:200], X[200:], y[200:], seed=1, grid=grid, n_classes=2)
    assert r.accuracy > 0.9


def test_verdict_rules():
    def mk(vals):
        def f(t, b, clf, a, ref):
            m = vals.get((t, b, clf, ref))
            return None if m is None else {"mean": m, "lo": m - 0.01, "hi": m + 0.01}
        return f
    cells = [(t, b, "lr") for t in ("sst2", "agnews") for b in (256, 1024)]
    go = {(t, b, "lr", r): 0.05 for t, b, _ in cells for r in ("bag", "young")}
    assert TR.verdict(mk(go), ["lr"], all_cells=cells)["verdict"] == "GO"
    nogo = {(t, b, "lr", r): -0.05 for t, b, _ in cells for r in ("bag", "young")}
    assert TR.verdict(mk(nogo), ["lr"], all_cells=cells)["verdict"] == "NO-GO"
    grey = dict(nogo)
    grey[("sst2", 256, "lr", "young")] = 0.05
    assert TR.verdict(mk(grey), ["lr"], all_cells=cells)["verdict"] == "GREY"


def test_lr_column_restriction_is_the_same_model():
    """Fitting on the used columns only equals fitting on all columns (unused columns get weight 0)."""
    sk = pytest.importorskip("sklearn.linear_model")
    import scipy.sparse as sp
    import warnings

    rng = np.random.default_rng(3)
    n, d = 120, 5000
    rows, cols = [], []
    for i in range(n):
        k = rng.integers(3, 12)
        rows += [i] * int(k)
        cols += rng.integers(0, 300, size=int(k)).tolist()      # columns 300.. never used
    X = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, d))
    X.sum_duplicates()
    y = (np.asarray(X[:, :150].sum(axis=1)).ravel() > np.asarray(X[:, 150:300].sum(axis=1)).ravel()).astype(int)
    Xtr, ytr, Xte = X[:80], y[:80], X[80:]
    model = TR._fit_lr(Xtr, ytr, 10.0, dense=False, max_iter=500)
    p_restricted = TR._predict_lr(model, Xte, dense=False)
    full = sk.LogisticRegression(C=10.0, solver="lbfgs", max_iter=500, tol=1e-4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full.fit(Xtr, ytr)
    assert np.array_equal(p_restricted, full.predict(Xte))
    w_full = full.coef_.ravel()
    unused = np.setdiff1d(np.arange(d), TR.used_columns(Xtr))
    assert np.all(w_full[unused] == 0.0)
    assert np.allclose(w_full[TR.used_columns(Xtr)], model[1].coef_.ravel(), atol=1e-6)
