"""Adversarial review tests of the CVTM stage-1 library (src/tmlm/cvtm/, docs/11 section 11, v0.3).

Every reference below is written from the definitions, independently of the code under test:

- Kneser-Ney (11.2): a textbook interpolated modified Kneser-Ney over Python dicts (raw counts at the
  top order, N1+(. g) continuation counts below, discounts from the counts of counts, uniform floor
  over the V real tokens), checked against `KNModel.prob`, `fit_kn` (numpy and CuPy) and the vote form
  of `SuffixClauses` at EVERY (position, class) pair of a held-out stream that contains a token never
  seen in training; the vote form must also normalise over all classes.
- Count pruning with the backoff-mass rule (`fit_kn(min_counts=...)`, default prune_mass="backoff"):
  a dict reference of "the unpruned model with the pruned followers removed and their mass moved to
  the interpolation weight".
- Document clauses (11.3): literal tuple counts and a common-suffix dynamic program for the pointer,
  on adversarial documents (runs, periods 2, 3 and 7, Fibonacci and Thue-Morse words, empty and short
  documents) with windows placed around every document edge and several pointer ranges. Batch and
  streaming must equal the reference AND report zero fallbacks: on these inputs a fallback would be a
  logic error (checks failing), not a hash collision (probability below 1e-30 per pair).
- Leakage: every feature at p is unchanged when any token at a position >= p, or of another document,
  changes (the gold count g_m(p) excepted, which is defined through tok[p]).
- Combination (11.4): the full distribution against the formula with every component active; bucket
  ids against (k*, m*, c, l) computed from the references; MAP-EM against a loop implementation, and
  bit-identical across two fresh processes.

No test reads data/, results/ or the test split.
"""

from __future__ import annotations

import hashlib
import inspect
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest

from tmlm.cvtm import doc_clauses as dc
from tmlm.cvtm import mixture as mx
from tmlm.cvtm.kn_gpu import fit_kn, model_arrays
from tmlm.cvtm.suffix import Q16, SuffixClauses
from tmlm.eval.baselines.ngram import KNModel

ROOT = Path(__file__).resolve().parents[1]


def _gpu_ok() -> bool:
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


GPU = _gpu_ok()
BACKENDS = ["numpy"] + (["cupy"] if GPU else [])
# 2026-10-25 (TLM-B3): the cupy case is a GPU test; a test that loops over BACKENDS uses the GPU
# exactly when one is visible, so it carries the gpu marker then (make test hides the GPU).
BACKEND_PARAMS = ["numpy"] + ([pytest.param("cupy", marks=pytest.mark.gpu)] if GPU else [])
gpu_when_visible = pytest.mark.gpu if GPU else (lambda f: f)


# ------------------------------------------------------------------ corpora
def chain_docs(seed: int, lengths, used: int, conc: float = 0.3) -> list[list[int]]:
    """Documents from a skewed first-order Markov chain over the token ids 0..used-1."""
    rng = np.random.default_rng(seed)
    trans = rng.dirichlet(np.full(used, conc), size=used)
    docs = []
    for n in lengths:
        d: list[int] = []
        if n:
            d.append(int(rng.integers(used)))
            for _ in range(int(n) - 1):
                d.append(int(rng.choice(used, p=trans[d[-1]])))
        docs.append(d)
    return docs


def to_stream(docs) -> tuple[np.ndarray, np.ndarray]:
    tok = np.array([t for d in docs for t in d], dtype=np.int64)
    ds = np.concatenate([[0], np.cumsum([len(d) for d in docs])]).astype(np.int64)
    return tok, ds


def left_contexts(tok: np.ndarray, ds: np.ndarray, m: int, bos: int) -> np.ndarray:
    """[n, m] BOS-padded in-document left contexts, built from the definition (last column nearest)."""
    out = np.full((tok.shape[0], m), bos, dtype=np.int64)
    for d in range(ds.shape[0] - 1):
        a, b = int(ds[d]), int(ds[d + 1])
        for p in range(a, b):
            for j in range(m):
                q = p - m + j
                if q >= a:
                    out[p, j] = tok[q]
    return out


# ------------------------------------------------------------------ textbook Kneser-Ney
class BruteKN:
    """Interpolated modified Kneser-Ney (Chen and Goodman 1999) over dicts, order n, V real tokens,
    every document padded with n-1 BOS (id V, never a target)."""

    def __init__(self, docs, order: int, vocab: int) -> None:
        n = order
        self.n, self.V = n, vocab
        raw = [None] + [Counter() for _ in range(n)]
        for d in docs:
            s = [vocab] * (n - 1) + list(d)
            for i in range(n - 1, len(s)):
                for k in range(1, n + 1):
                    raw[k][tuple(s[i - k + 1:i + 1])] += 1
        self.raw = raw
        cnt: list = [None] * (n + 1)
        cnt[n] = dict(raw[n])
        for k in range(1, n):
            c: Counter = Counter()
            for g, v in raw[k + 1].items():
                if v > 0:
                    c[g[1:]] += 1                  # N1+(. g): distinct left extensions in the raw counts
            cnt[k] = dict(c)
        self.cnt = cnt
        self.D = [None] + [self._discounts(list(cnt[k].values())) for k in range(1, n + 1)]
        self.fol: list = [None] * (n + 1)
        for k in range(1, n + 1):
            f: dict = defaultdict(dict)
            for g, v in cnt[k].items():
                if v > 0:
                    f[g[:-1]][g[-1]] = v
            self.fol[k] = f

    @staticmethod
    def _discounts(vals) -> tuple[float, float, float]:
        n1, n2, n3, n4 = (sum(1 for v in vals if v == i) for i in (1, 2, 3, 4))
        y = n1 / (n1 + 2 * n2) if n1 + 2 * n2 > 0 else 0.0
        d1 = 1.0 - 2.0 * y * (n2 / n1 if n1 else 0.0)
        d2 = 2.0 - 3.0 * y * (n3 / n2 if n2 else 0.0)
        d3 = 3.0 - 4.0 * y * (n4 / n3 if n3 else 0.0)
        return min(max(d1, 0.0), 1.0), min(max(d2, 0.0), 2.0), min(max(d3, 0.0), 3.0)

    def disc(self, k: int, c: int) -> float:
        d1, d2, d3 = self.D[k]
        return 0.0 if c <= 0 else d1 if c == 1 else d2 if c == 2 else d3

    def total(self, u: tuple) -> int:
        """Follower total of context u at level len(u) + 1."""
        return sum(self.fol[len(u) + 1].get(u, {}).values())

    def p(self, w: int, u: tuple, keep_min: dict | None = None) -> float:
        """P_k(w | u), k = len(u) + 1. With keep_min: the followers of RAW count below keep_min[k]
        are removed and their count mass is moved to the interpolation weight (backoff rule)."""
        k = len(u) + 1
        f = self.fol[k].get(u, {})
        tot = sum(f.values())
        lower = 1.0 / self.V if k == 1 else self.p(w, u[1:], keep_min)
        if tot == 0:
            return lower
        thr = 0 if keep_min is None else keep_min.get(k, 0)
        kept = {x: c for x, c in f.items() if self.raw[k][u + (x,)] >= thr}
        mass = sum(self.disc(k, c) for c in kept.values()) + sum(c for x, c in f.items() if x not in kept)
        c = kept.get(w, 0)
        return (max(c - self.disc(k, c), 0.0) + mass * lower) / tot

    def k_star(self, u: tuple) -> int:
        """The longest k <= len(u) whose length-k suffix of u has a positive follower total."""
        best = 0
        for k in range(1, len(u) + 1):
            if self.total(tuple(u[len(u) - k:])) > 0:
                best = k
        return best


def kn_case(order: int, vocab: int, seed: int):
    lengths = [0, 1, 2, 3, 4, 5, 9, 30, 60, 0, 17, 45, 80, 3]
    train = chain_docs(seed, lengths, used=vocab - 1)          # token vocab-1 never occurs in training
    held = chain_docs(seed + 50, [0, 1, 4, 12, 40, 25], used=vocab)
    return train, held


@pytest.mark.parametrize("order,vocab,seed", [(2, 5, 1), (3, 6, 2), (5, 5, 3), (5, 7, 4)])
@gpu_when_visible
def test_vote_form_equals_textbook_kneser_ney_at_every_class(order, vocab, seed):
    train, held = kn_case(order, vocab, seed)
    bk = BruteKN(train, order, vocab)
    tok, ds = to_stream(train)
    etok, eds = to_stream(held)
    ctx = left_contexts(etok, eds, order - 1, vocab)
    q = ctx.shape[0]
    ref = np.array([[bk.p(w, tuple(int(x) for x in row)) for w in range(vocab)] for row in ctx])
    assert np.allclose(ref.sum(axis=1), 1.0, rtol=0, atol=1e-12)          # the reference itself is proper
    qq = np.repeat(np.arange(q), vocab)
    ww = np.tile(np.arange(vocab), q)
    knr = KNModel(order, vocab).fit(tok, ds)
    assert np.allclose(knr.prob(ctx[qq], ww).reshape(q, vocab), ref, rtol=1e-12, atol=0.0)
    kctx, kvalid = knr.contexts(etok, eds, np.arange(q))
    assert np.array_equal(kctx, ctx) and np.all(kvalid == order - 1)
    for backend in BACKENDS:
        kn = fit_kn(tok, ds, order, vocab=vocab, backend=backend)
        sc = SuffixClauses.from_kn(kn, backend=backend)
        lp = sc.logprob(ctx[qq], ww).reshape(q, vocab)
        assert float(np.max(np.abs(lp - np.log(ref)))) <= 1e-12, backend
        assert np.allclose(np.exp(lp).sum(axis=1), 1.0, rtol=0, atol=1e-12)
        lq = sc.logprob_q(ctx[qq], ww).reshape(q, vocab)
        bound = sc.rounding_bound(ctx[qq], ww).reshape(q, vocab)
        assert np.all(np.abs(lq / Q16 - lp) <= bound + 1e-12)
        # a clause fires exactly when its context has a positive follower total
        fired = sc.fired(ctx)
        for r, row in enumerate(ctx):
            for k in range(1, order):
                u = tuple(int(x) for x in row[order - 1 - k:])
                assert (fired[r, k - 1] >= 0) == (bk.total(u) > 0), (r, k, u)
        # k* and P_KN(gold) of the mixture helper
        pg, ks = mx.kn_gold_and_depth(kn, etok, eds)
        assert np.allclose(pg, ref[np.arange(q), etok], rtol=1e-12, atol=0.0)
        assert ks.tolist() == [bk.k_star(tuple(int(x) for x in row)) for row in ctx]


# ------------------------------------------------------------------ pruning, backoff-mass rule
@pytest.mark.parametrize("m,seed", [({3: 2, 4: 2, 5: 2}, 14), ({3: 3, 4: 3, 5: 3}, 20), ({4: 2, 5: 3}, 21),
                                    ({5: 2}, 22), ({3: 2, 4: 2, 5: 2}, 38)])
@gpu_when_visible
def test_backoff_pruning_equals_its_definition(m, seed):
    order, vocab = 5, 6
    lengths = [0, 1, 3, 5, 40, 80, 120, 150, 60, 7, 90]
    train = chain_docs(seed, lengths, used=vocab - 1, conc=0.3)
    held = chain_docs(seed + 70, [2, 30, 50, 11], used=vocab, conc=0.3)
    bk = BruteKN(train, order, vocab)
    tok, ds = to_stream(train)
    # precondition: every unpruned discount is positive. On tiny skewed corpora the clamp of modified
    # Kneser-Ney can give D = 0 (then even the UNPRUNED model has contexts with gamma = 0, which the vote
    # form refuses); the seeds above were chosen to avoid that, and this line keeps them honest.
    assert all(d > 0 for k in range(1, order + 1) for d in bk.D[k])
    etok, eds = to_stream(held)
    ctx = left_contexts(etok, eds, order - 1, vocab)
    q = ctx.shape[0]
    qq = np.repeat(np.arange(q), vocab)
    ww = np.tile(np.arange(vocab), q)
    ref = np.array([[bk.p(w, tuple(int(x) for x in row), m) for w in range(vocab)] for row in ctx])
    assert np.allclose(ref.sum(axis=1), 1.0, rtol=0, atol=1e-12)
    full = fit_kn(tok, ds, order, vocab=vocab, backend="numpy")
    for backend in BACKENDS:
        kn = fit_kn(tok, ds, order, vocab=vocab, backend=backend, min_counts=m)       # defaults: full, backoff
        assert kn.fit_info["pruning"]["prune_cont"] == "full" and kn.fit_info["pruning"]["prune_mass"] == "backoff"
        assert np.allclose(kn.prob(ctx[qq], ww).reshape(q, vocab), ref, rtol=1e-12, atol=0.0), backend
        assert np.allclose(kn.dist(ctx).sum(axis=1), 1.0, rtol=0, atol=1e-12)
        sc = SuffixClauses.from_kn(kn, backend=backend)                              # gamma > 0 everywhere
        lp = sc.logprob(ctx[qq], ww).reshape(q, vocab)
        assert float(np.max(np.abs(lp - np.log(ref)))) <= 1e-12
        # the same kept inventory as the renormalising variant, smaller than the unpruned model
        ren = fit_kn(tok, ds, order, vocab=vocab, backend=backend, min_counts=m, prune_mass="renormalize")
        for k in range(1, order + 1):
            assert np.array_equal(kn.keys[k], ren.keys[k])
            for lv, lf in ((kn.raw[k], full.raw[k]), (kn.cont[k], full.cont[k])):
                if lv is not None:                                                    # unpruned discounts
                    assert [x.hex() for x in lv.d] == [x.hex() for x in lf.d]
        assert any(kn.n_types[k] < full.n_types[k] for k in m)
        assert sc.nbytes() < SuffixClauses.from_kn(full, backend="numpy").nbytes()
        assert any(name.endswith(".extra") for name in model_arrays(kn))


def test_backoff_pruning_with_zero_thresholds_is_the_unpruned_model():
    train, _ = kn_case(5, 6, 9)
    tok, ds = to_stream(train)
    ref = KNModel(5, 6).fit(tok, ds)
    kn = fit_kn(tok, ds, 5, vocab=6, backend="numpy", min_counts={3: 0, 4: 0, 5: 0})
    ctx = left_contexts(tok, ds, 4, 6)
    assert np.array_equal(kn.dist(ctx), ref.dist(ctx))
    assert not any(name.endswith(".extra") for name in model_arrays(kn))


# ------------------------------------------------------------------ document clauses
def fibonacci_word(n: int) -> list[int]:
    a, b = [0], [0, 1]
    while len(b) < n:
        a, b = b, b + a
    return b[:n]


def thue_morse(n: int) -> list[int]:
    return [bin(i).count("1") % 2 for i in range(n)]


def adversarial_docs() -> list[list[int]]:
    rng = np.random.default_rng(1234)
    period7 = [i % 7 for i in range(140)]
    for i in (20, 51, 52, 99):
        period7[i] = (period7[i] + 3) % 7
    return [
        [], [3], [0] * 4, [0] * 5, [0] * 6, [1] * 150, [0, 1] * 70, ([2, 0, 1] * 30)[:89],
        fibonacci_word(150), thue_morse(128), period7, [], [4] * 65, [4] * 66, [4] * 70,
        [int(x) for x in rng.integers(0, 2, 120)], [int(x) for x in rng.integers(0, 3, 100)], [5, 5],
    ]


class BruteDoc:
    """Document-clause features straight from docs/11 section 11.3."""

    def __init__(self, tok: np.ndarray, ds: np.ndarray) -> None:
        self.tok = [int(x) for x in tok]
        self.ds = [int(x) for x in ds]
        self.n = len(self.tok)
        # lcs[d][i][j]: number of equal tokens immediately left of positions a+i and a+j, inside the document
        self.lcs = []
        for d in range(len(self.ds) - 1):
            a, b = self.ds[d], self.ds[d + 1]
            size = b - a
            t = self.tok[a:b]
            lcs = np.zeros((size + 1, size + 1), dtype=np.int64)
            for i in range(1, size + 1):
                for j in range(1, size + 1):
                    if t[i - 1] == t[j - 1]:
                        lcs[i, j] = lcs[i - 1, j - 1] + 1
            self.lcs.append(lcs)

    def window(self, a: int, p: int, window: int | None) -> range:
        return range(a if window is None else max(a, p - window), p)

    def doc_of(self, p: int) -> int:
        return int(np.searchsorted(self.ds, p, side="right")) - 1

    def successors(self, p: int, m: int, window: int | None) -> Counter:
        """tok[q] over the q in the window of p whose m-token context equals that of p (both in-document)."""
        d = self.doc_of(p)
        a = self.ds[d]
        t = self.tok
        if p - m < a:
            return Counter()
        c = tuple(t[p - m:p])
        return Counter(t[q] for q in self.window(a, p, window) if q - m >= a and tuple(t[q - m:q]) == c)

    def features(self, orders, window, ptr_min, ptr_max):
        n = self.n
        cnt_n = np.zeros((len(orders), n), dtype=np.int64)
        cnt_g = np.zeros((len(orders), n), dtype=np.int64)
        for p in range(n):
            for i, m in enumerate(orders):
                s = self.successors(p, m, window)
                cnt_n[i, p] = sum(s.values())
                cnt_g[i, p] = s.get(self.tok[p], 0)
        return cnt_n, cnt_g

    def pointer(self, window, ptr_min, ptr_max):
        n = self.n
        length = np.zeros(n, dtype=np.int64)
        pos = np.full(n, -1, dtype=np.int64)
        for d in range(len(self.ds) - 1):
            a, b = self.ds[d], self.ds[d + 1]
            lcs = self.lcs[d]
            for p in range(a, b):
                best, bq = 0, -1
                for q in self.window(a, p, window):
                    v = min(int(lcs[p - a, q - a]), ptr_max)
                    if v >= best and v > 0:                  # ties: the most recent q wins
                        best, bq = v, q
                if best >= ptr_min:
                    length[p], pos[p] = best, bq
        return length, pos


@pytest.fixture(scope="module")
def adversarial_stream():
    tok, ds = to_stream(adversarial_docs())
    return tok, ds, BruteDoc(tok, ds)


WINDOWS = [None, 0, 1, 4, 5, 6, 63, 64, 65, 66, 69, 88, 89, 127, 128, 149, 150, 151, 1000]
PTR_RANGES = [(5, 64), (1, 1), (1, 3), (2, 6), (5, 5), (64, 64)]


def test_doc_features_equal_the_definition_at_document_edges(adversarial_stream):
    tok, ds, bd = adversarial_stream
    orders = (0, 1, 2, 3, 4)
    for window in WINDOWS:
        ref_n, ref_g = bd.features(orders, window, 5, 64)
        for ptr_min, ptr_max in PTR_RANGES:
            ref_l, ref_q = bd.pointer(window, ptr_min, ptr_max)
            ref_c = np.where(ref_q >= 0, tok[np.maximum(ref_q, 0)], -1)
            for fn in (dc.doc_features, dc.doc_features_stream):
                f = fn(tok, ds, orders, window, ptr_min, ptr_max)
                tag = (fn.__name__, window, ptr_min, ptr_max)
                assert f.n_fallback == 0, tag                 # a fallback here is a logic error, not a collision
                assert np.array_equal(f.n, ref_n) and np.array_equal(f.g, ref_g), tag
                assert np.array_equal(f.ptr_len, ref_l), tag
                assert np.array_equal(f.ptr_pos, ref_q), tag
                assert np.array_equal(f.ptr_cand, ref_c), tag


@pytest.mark.parametrize("orders", [(4,), (2, 4), (0, 3), (1,)])
def test_doc_feature_order_subsets(adversarial_stream, orders):
    tok, ds, bd = adversarial_stream
    for window in (None, 5, 66):
        ref_n, ref_g = bd.features(orders, window, 5, 64)
        for fn in (dc.doc_features, dc.doc_features_stream):
            f = fn(tok, ds, orders, window)
            assert f.n_fallback == 0
            assert np.array_equal(f.n, ref_n) and np.array_equal(f.g, ref_g), (fn.__name__, orders, window)


def test_doc_candidates_equal_the_definition(adversarial_stream):
    tok, ds, bd = adversarial_stream
    rng = np.random.default_rng(3)
    positions = np.sort(rng.choice(tok.shape[0], size=120, replace=False))
    for window in (None, 1, 6, 65, 150):
        cands = dc.doc_candidates(tok, ds, positions, window=window)
        for m in range(5):
            for r, p in enumerate(positions):
                t, c = cands[m].row(r)
                ref = bd.successors(int(p), m, window)
                assert dict(zip(t.tolist(), c.tolist())) == dict(ref), (window, m, int(p))
                assert t.tolist() == sorted(ref)


def test_no_leakage_from_the_future_or_other_documents(adversarial_stream):
    tok, ds, _ = adversarial_stream
    rng = np.random.default_rng(11)
    lengths = np.diff(ds)
    for trial in range(8):
        d = int(rng.choice(np.flatnonzero(lengths >= 20)))
        a, b = int(ds[d]), int(ds[d + 1])
        p = int(rng.integers(a + 10, b))
        tok2 = tok.copy()
        tok2[p:] = rng.integers(0, 7, size=tok.shape[0] - p)          # the gold, the rest of the doc, later docs
        tok2[:a] = rng.integers(0, 7, size=a)                          # every earlier document
        window = [None, 3, 17, 64][trial % 4]
        for fn in (dc.doc_features, dc.doc_features_stream):
            f1 = fn(tok, ds, window=window)
            f2 = fn(tok2, ds, window=window)
            sl = slice(a, p)                                           # positions before p: nothing may change
            assert np.array_equal(f1.n[:, sl], f2.n[:, sl]) and np.array_equal(f1.g[:, sl], f2.g[:, sl])
            assert np.array_equal(f1.ptr_len[a:p + 1], f2.ptr_len[a:p + 1])
            assert np.array_equal(f1.ptr_pos[a:p + 1], f2.ptr_pos[a:p + 1])
            assert np.array_equal(f1.ptr_cand[a:p + 1], f2.ptr_cand[a:p + 1])
            assert np.array_equal(f1.n[:, p], f2.n[:, p])              # at p: all but the gold count
        c1 = dc.doc_candidates(tok, ds, [p], window=window)
        c2 = dc.doc_candidates(tok2, ds, [p], window=window)
        for m in range(5):
            assert all(np.array_equal(x, y) for x, y in zip(c1[m].row(0), c2[m].row(0)))
    # the suffix-clause side: contexts, fired clauses and k* at p do not see tok[p:]
    train = chain_docs(21, [30, 50, 0, 80], used=7)
    ttok, tds = to_stream(train)
    kn = KNModel(5, 7).fit(ttok, tds)
    sc = SuffixClauses.from_kn(kn, backend="numpy")
    pos = np.arange(tok.shape[0])
    for p in (int(ds[5]) + 7, int(ds[8]) + 40):
        tok2 = tok.copy()
        tok2[p:] = (tok2[p:] + 1) % 7
        c1, v1 = sc.contexts(tok, ds, pos[: p + 1])
        c2, v2 = sc.contexts(tok2, ds, pos[: p + 1])
        assert np.array_equal(c1, c2) and np.array_equal(v1, v2)
        assert np.array_equal(sc.fired(c1, v1), sc.fired(c2, v2))
        k1 = mx.kn_gold_and_depth(kn, tok, ds, pos[: p + 1])[1]
        k2 = mx.kn_gold_and_depth(kn, tok2, ds, pos[: p + 1])[1]
        assert np.array_equal(k1, k2)


# ------------------------------------------------------------------ combination (11.4)
def _count_bucket(n: int) -> int:
    for idx, (lo, hi) in enumerate(((1, 1), (2, 3), (4, 7), (8, 15))):
        if lo <= n <= hi:
            return idx
    return 4


def _pointer_bucket(length: int) -> int:
    if length < 5:
        return 0
    return 1 if length <= 8 else 2 if length <= 16 else 3


@pytest.fixture(scope="module")
def mixture_case():
    return make_mixture_case()


def make_mixture_case():
    vocab, order = 6, 5
    train = chain_docs(31, [40, 70, 0, 120, 9, 55], used=vocab)
    bk = BruteKN(train, order, vocab)
    ttok, tds = to_stream(train)
    kn = KNModel(order, vocab).fit(ttok, tds)
    base_doc = [0, 3, 1, 4, 2, 5, 1]
    held = [base_doc * 9 + [2, 2, 0], chain_docs(32, [70], used=vocab)[0], [1] * 30]
    etok, eds = to_stream(held)
    return vocab, order, bk, kn, etok, eds, BruteDoc(etok, eds)


def test_full_distribution_equals_the_formula_with_every_component_active(mixture_case):
    vocab, order, bk, kn, etok, eds, bd = mixture_case
    feats = dc.doc_features(etok, eds)
    ctx = left_contexts(etok, eds, order - 1, vocab)
    base = kn.dist(ctx)
    act = mx.active_mask(mx.counts_matrix(feats), feats.ptr_len)
    rows = np.flatnonzero(act.all(axis=1))
    assert rows.shape[0] >= 30                                   # the case really has every component active
    rng = np.random.default_rng(5)
    cands = dc.doc_candidates(etok, eds, rows)
    for comps in (None, mx.CACHE_COMPONENTS, ("S", "D2", "P")):
        cm = mx.component_mask(comps)
        for r, p in enumerate(rows):
            p = int(p)
            w = rng.dirichlet(np.ones(7))
            succ = [bd.successors(p, m, None) for m in range(5)]
            cand = int(feats.ptr_cand[p])
            num = w[0] * base[p].copy()
            den = w[0]
            for m in range(5):
                if cm[1 + m]:
                    nm = sum(succ[m].values())
                    for t, c in succ[m].items():
                        num[t] += w[1 + m] * c / nm
                    den += w[1 + m]
            if cm[6]:
                num[cand] += w[6]
                den += w[6]
            ref = num / den
            doc_rows = [cands[m].row(r) for m in range(5)]
            full = mx.full_distribution(base[p], doc_rows, cand, w, comps)
            assert np.allclose(full, ref, rtol=1e-14, atol=1e-16)
            assert abs(float(full.sum()) - 1.0) <= 1e-12
            # gold entry: bit-identical to gold_prob on the same inputs
            pc = np.zeros((1, 7))
            pc[0, 0] = base[p, etok[p]]
            for m in range(5):
                pc[0, 1 + m] = succ[m].get(int(etok[p]), 0) / sum(succ[m].values())
            pc[0, 6] = float(cand == etok[p])
            g = mx.gold_prob(pc, np.ones((1, 7), bool), np.zeros(1, np.int64), w[None, :], comps)
            assert float(g[0]) == float(full[etok[p]])
            # vote form: P = P_KN exp(r), default vote ln(l_S / sum of active weights)
            dv, ct, cv = mx.residual_votes(base[p], doc_rows, cand, w, comps)
            rv = np.full(vocab, dv)
            rv[ct] = cv
            assert np.allclose(base[p] * np.exp(rv), full, rtol=1e-13, atol=0.0)


def test_buckets_equal_the_definition(mixture_case):
    vocab, order, bk, kn, etok, eds, bd = mixture_case
    feats = dc.doc_features(etok, eds)
    ctx = left_contexts(etok, eds, order - 1, vocab)
    ks = mx.kn_gold_and_depth(kn, etok, eds)[1]
    assert ks.tolist() == [bk.k_star(tuple(int(x) for x in row)) for row in ctx]
    ref_n, _ = bd.features((0, 1, 2, 3, 4), None, 5, 64)
    ref_l, _ = bd.pointer(None, 5, 64)
    for comps in (None, mx.CACHE_COMPONENTS):
        lay = mx.BucketLayout.of(comps)
        bucket, nb = mx.bucket_ids(ks, mx.counts_matrix(feats), feats.ptr_len, comps)
        assert nb == (520 if comps is None else 30)
        act = mx.active_mask(mx.counts_matrix(feats), feats.ptr_len, comps)
        for p in range(etok.shape[0]):
            fire = [m for m in lay.d_orders if ref_n[m, p] >= 1]
            m_star = max(fire) if fire else None
            c = _count_bucket(int(ref_n[m_star, p])) if fire else None
            l_b = _pointer_bucket(int(ref_l[p])) if lay.has_pointer else 0
            assert lay.decode(int(bucket[p])) == (int(ks[p]), m_star, c, l_b), p
            assert np.array_equal(lay.active_components(int(bucket[p])), act[p]), p


def _loop_map_em(p, act, bucket, nb, iters, prior):
    """MAP-EM of docs/11 11.4 written with explicit loops (the pooled MM step of mixture.py's
    docstring, then per bucket EM with prior * pi_i / pi(A_b) pseudo-counts)."""
    n, k = len(p), len(p[0])
    ever = [any(act[r][i] for r in range(n)) for i in range(k)]
    w = [1.0 / sum(ever) if ever[i] else 0.0 for i in range(k)]
    for _ in range(iters):
        nk = [0.0] * k
        ck = [0.0] * k
        for r in range(n):
            s = sum(w[i] for i in range(k) if act[r][i])
            tot = sum(w[i] * p[r][i] for i in range(k) if act[r][i])
            for i in range(k):
                if act[r][i]:
                    nk[i] += w[i] * p[r][i] / tot
                    ck[i] += 1.0 / s
        w = [nk[i] / ck[i] if ck[i] > 0 else 0.0 for i in range(k)]
        z = sum(w)
        w = [x / z for x in w]
    out = np.tile(np.array(w), (nb, 1))
    for b in range(nb):
        rows = [r for r in range(n) if bucket[r] == b]
        if not rows:
            continue
        A = [bool(act[rows[0]][i]) for i in range(k)]
        pa = sum(w[i] for i in range(k) if A[i])
        a = [prior * w[i] / pa if A[i] else 0.0 for i in range(k)]
        v = [1.0 / sum(A) if A[i] else 0.0 for i in range(k)]
        for _ in range(iters):
            nb_ = [0.0] * k
            for r in rows:
                tot = sum(v[i] * p[r][i] for i in range(k) if A[i])
                for i in range(k):
                    if A[i]:
                        nb_[i] += v[i] * p[r][i] / tot
            v = [(nb_[i] + a[i]) / (len(rows) + prior) if A[i] else 0.0 for i in range(k)]
        out[b] = v
    return out


def _em_rows(seed: int, n: int):
    rng = np.random.default_rng(seed)
    n_counts = np.zeros((5, n), dtype=np.int64)
    top = rng.integers(-1, 5, n)                                    # highest firing order (nested by construction)
    for m in range(5):
        n_counts[m] = np.where(top >= m, rng.integers(1, 40, n), 0)
    ptr = np.where((top == 4) & (rng.random(n) < 0.5), rng.integers(5, 40, n), 0)
    ks = rng.integers(0, 5, n)
    act = mx.active_mask(n_counts, ptr)
    p = np.where(act, rng.random((n, 7)) ** 3, 0.0)
    p[:, 0] = rng.random(n) * 0.5 + 1e-3
    bucket, nb = mx.bucket_ids(ks, n_counts, ptr)
    return p, act, bucket, nb


def test_map_em_equals_a_loop_implementation():
    p, act, bucket, nb = _em_rows(7, 400)
    got = mx.fit_map_em(p, act, bucket, iters=12, prior=16.0, n_buckets=nb)
    ref = _loop_map_em(p.tolist(), act.tolist(), bucket.tolist(), nb, 12, 16.0)
    assert np.allclose(got, ref, rtol=1e-11, atol=1e-14)
    assert np.allclose(got.sum(axis=1), 1.0, rtol=0, atol=1e-12)
    # the gold probability is the renormalised mixture over the active components
    g = mx.gold_prob(p, act, bucket, got)
    wb = got[bucket]
    assert np.allclose(g, (wb * p * act).sum(axis=1) / (wb * act).sum(axis=1), rtol=1e-14, atol=0.0)


_EM_SCRIPT = """
import hashlib
import numpy as np
from tmlm.cvtm import mixture as mx
{helper}
p, act, bucket, nb = _em_rows(17, 30000)
w = mx.fit_map_em(p, act, bucket, iters=50, prior=16.0, n_buckets=nb)
print(hashlib.sha256(np.ascontiguousarray(w).tobytes()).hexdigest())
"""


def test_map_em_is_bit_identical_across_processes():
    p, act, bucket, nb = _em_rows(17, 30000)
    here = hashlib.sha256(np.ascontiguousarray(mx.fit_map_em(p, act, bucket, iters=50, prior=16.0,
                                                             n_buckets=nb)).tobytes()).hexdigest()
    digests = []
    for hashseed in ("0", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=hashseed, PYTHONPATH=str(ROOT / "src"),
                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        script = _EM_SCRIPT.format(helper=inspect.getsource(_em_rows))     # no import of this module (no CuPy)
        out = subprocess.run([sys.executable, "-c", script], env=env,
                             capture_output=True, text=True, timeout=120, check=True)
        digests.append(out.stdout.strip().splitlines()[-1])
    assert digests == [here, here]
