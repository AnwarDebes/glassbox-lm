"""Document clauses of the CVTM (docs/11 section 11.3): the batch features equal a brute-force scan
and the streaming (numba) features on random multi-document streams, windows {None, 1, 3, 17};
a planted repetition; forced hash collisions; the candidate CSR rows; argument checks."""

from __future__ import annotations

import numpy as np
import pytest

from tmlm.cvtm.doc_clauses import (EDGE, DocFeatures, doc_candidates, doc_features, doc_features_stream,
                                   ns_per_token)

WINDOWS = (None, 1, 3, 17)


# ------------------------------------------------------------------ brute force (independent)
def _ctx(tok, a, x, m):
    return tuple(int(tok[x - i]) if x - i >= a else EDGE for i in range(1, m + 1))


def brute_force(tok, doc_start, orders, window, ptr_min, ptr_max):
    tok = np.asarray(tok, dtype=np.int64)
    n = tok.shape[0]
    cnt_n = np.zeros((len(orders), n), np.int64)
    cnt_g = np.zeros((len(orders), n), np.int64)
    plen = np.zeros(n, np.int64)
    ppos = np.full(n, -1, np.int64)
    for d in range(len(doc_start) - 1):
        a, b = int(doc_start[d]), int(doc_start[d + 1])
        for p in range(a, b):
            lo = a if window is None else max(a, p - window)
            for i, m in enumerate(orders):
                cp = _ctx(tok, a, p, m)
                for q in range(lo, p):
                    if _ctx(tok, a, q, m) == cp:
                        cnt_n[i, p] += 1
                        cnt_g[i, p] += int(tok[q] == tok[p])
            best_l, best_q = 0, -1
            for q in range(p - 1, lo - 1, -1):          # most recent first
                length = 0
                while length < ptr_max and q - 1 - length >= a and tok[p - 1 - length] == tok[q - 1 - length]:
                    length += 1
                if length > best_l:
                    best_l, best_q = length, q
            if best_l >= ptr_min:
                plen[p], ppos[p] = best_l, best_q
    return cnt_n, cnt_g, plen, ppos


def random_stream(seed, n_docs=6, max_len=160, vocab=5, empty_doc=True):
    rng = np.random.default_rng(seed)
    lengths = rng.integers(1, max_len, size=n_docs)
    if empty_doc:
        lengths[n_docs // 2] = 0
    doc_start = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    n = int(doc_start[-1])
    # a mix of random tokens and copied spans so that long pointers exist
    tok = rng.integers(0, vocab, size=n)
    for _ in range(n // 20):
        src = int(rng.integers(0, max(1, n - 30)))
        dst = int(rng.integers(0, max(1, n - 30)))
        span = int(rng.integers(3, 25))
        tok[dst:dst + span] = tok[src:src + span].copy()
    return tok.astype(np.int64), doc_start


def assert_matches_brute(f: DocFeatures, tok, doc_start):
    bn, bg, bl, bq = brute_force(tok, doc_start, f.orders, f.window, f.ptr_min, f.ptr_max)
    np.testing.assert_array_equal(f.n, bn)
    np.testing.assert_array_equal(f.g, bg)
    np.testing.assert_array_equal(f.ptr_len, bl)
    np.testing.assert_array_equal(f.ptr_pos, bq)
    np.testing.assert_array_equal(f.ptr_cand, np.where(bq >= 0, np.asarray(tok)[np.maximum(bq, 0)], -1))


# ------------------------------------------------------------------ tests
@pytest.mark.parametrize("window", WINDOWS)
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_batch_and_stream_equal_brute_force(seed, window):
    tok, ds = random_stream(seed)
    for ptr_min, ptr_max in ((5, 64), (2, 7)):
        fb = doc_features(tok, ds, window=window, ptr_min=ptr_min, ptr_max=ptr_max)
        assert_matches_brute(fb, tok, ds)
        fs = doc_features_stream(tok, ds, window=window, ptr_min=ptr_min, ptr_max=ptr_max)
        assert fs.same_values(fb)
        assert fb.n_fallback == 0 and fs.n_fallback == 0


@pytest.mark.parametrize("window", WINDOWS)
def test_subset_orders_and_larger_vocab(window):
    tok, ds = random_stream(11, n_docs=5, max_len=220, vocab=40, empty_doc=False)
    fb = doc_features(tok, ds, orders=(1, 3), window=window, ptr_min=3, ptr_max=9)
    assert_matches_brute(fb, tok, ds)
    fs = doc_features_stream(tok, ds, orders=(1, 3), window=window, ptr_min=3, ptr_max=9)
    assert fs.same_values(fb) and fs.n_fallback == 0


@pytest.mark.parametrize("window", [None, 3, 17])
@pytest.mark.parametrize("base", [0, 1])
def test_forced_hash_collisions_stay_exact(window, base):
    """base 0 makes H_L depend on the last token only, base 1 on the token sum: collisions
    everywhere. Verification plus the exact scan must still give the exact pointer."""
    tok, ds = random_stream(5, n_docs=4, max_len=140, vocab=6)
    fb = doc_features(tok, ds, window=window, ptr_min=3, ptr_max=12, _hash_base=base)
    fs = doc_features_stream(tok, ds, window=window, ptr_min=3, ptr_max=12, _hash_base=base, _hash_base2=base)
    assert_matches_brute(fb, tok, ds)
    assert fs.same_values(fb)
    assert fb.n_fallback > 0 and fs.n_fallback > 0


def test_window_zero_fires_nothing():
    tok, ds = random_stream(7)
    for f in (doc_features(tok, ds, window=0), doc_features_stream(tok, ds, window=0)):
        assert not f.n.any() and not f.g.any() and not f.ptr_len.any()
        assert (f.ptr_pos == -1).all() and (f.ptr_cand == -1).all()


def test_planted_repetition():
    rng = np.random.default_rng(3)
    seg = rng.integers(100, 30000, size=30)            # distinct-looking 30-token segment
    filler = rng.integers(30000, 32000, size=50)
    doc0 = np.concatenate([seg, filler, seg])          # the segment twice in document 0
    doc1 = seg.copy()                                   # and once in document 1
    tok = np.concatenate([doc0, doc1]).astype(np.int64)
    ds = np.array([0, doc0.shape[0], tok.shape[0]], dtype=np.int64)
    for fn in (doc_features, doc_features_stream):
        f = fn(tok, ds)
        second = 30 + 50                                # start of the second copy
        for j in range(5, 30):                          # j tokens of the copy already seen
            p = second + j
            assert f.ptr_len[p] == j
            assert f.ptr_pos[p] == j
            assert f.ptr_cand[p] == seg[j]
            assert f.count(4)[p] >= 1 and f.gold_count(4)[p] >= 1
        assert f.ptr_len[second + 4] == 0               # 4 < ptr_min
        # document 1 never sees document 0
        assert not f.ptr_len[doc0.shape[0]:].any()
        np.testing.assert_array_equal(f.count(0)[doc0.shape[0]:], np.arange(30))
        # a window too short to reach the first copy switches the pointer off
        fw = fn(tok, ds, window=40)
        assert not fw.ptr_len[second:second + 30].any()


@pytest.mark.parametrize("window", [None, 3])
def test_candidates_match_brute_force(window):
    tok, ds = random_stream(9, n_docs=4, max_len=120, vocab=7)
    f = doc_features(tok, ds, window=window)
    rng = np.random.default_rng(0)
    pos = np.sort(rng.choice(tok.shape[0], size=60, replace=False))
    cands = doc_candidates(tok, ds, pos, window=window)
    doc_of = np.searchsorted(ds, np.arange(tok.shape[0]), side="right") - 1
    for m, csr in cands.items():
        for r, p in enumerate(pos):
            a = int(ds[doc_of[p]])
            lo = a if window is None else max(a, int(p) - window)
            want: dict[int, int] = {}
            for q in range(lo, int(p)):
                if _ctx(tok, a, q, m) == _ctx(tok, a, int(p), m):
                    want[int(tok[q])] = want.get(int(tok[q]), 0) + 1
            wt, wc = csr.row(r)
            assert list(wt) == sorted(want) and list(wc) == [want[k] for k in sorted(want)]
            assert int(wc.sum()) == int(f.count(m)[p])
            assert want.get(int(tok[p]), 0) == int(f.gold_count(m)[p])


def test_d_orders_nest_and_pointer_implies_d4():
    tok, ds = random_stream(4, vocab=4)
    f = doc_features(tok, ds)
    for m in range(1, 5):
        assert np.all((f.count(m) >= 1) <= (f.count(m - 1) >= 1))
    assert np.all((f.ptr_len >= 5) <= (f.count(4) >= 1))


def test_count_identity_n_equals_previous_g():
    """n_m(p) = g_(m-1)(p - 1) inside a document (the streaming path relies on it; the batch
    path computes both independently)."""
    tok, ds = random_stream(8, vocab=3)
    for window in (None, 2, 9):
        f = doc_features(tok, ds, window=window)
        doc_of = np.searchsorted(ds, np.arange(tok.shape[0]), side="right") - 1
        inner = np.flatnonzero(np.r_[False, doc_of[1:] == doc_of[:-1]])
        for m in range(1, 5):
            np.testing.assert_array_equal(f.count(m)[inner], f.gold_count(m - 1)[inner - 1])


def test_stream_stress_repetitive_streams():
    """Many small, highly repetitive streams (vocab 2 and 3, copied spans), all windows and pointer
    ranges: the streaming trie must agree with the batch path without any fallback."""
    for seed in range(120):
        rng = np.random.default_rng(1000 + seed)
        vocab = int(rng.choice([2, 3, 5]))
        lengths = rng.integers(0, 250, size=int(rng.integers(1, 5)))
        ds = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        n = int(ds[-1])
        tok = rng.integers(0, vocab, size=n)
        for _ in range(n // 15):
            src, dst = int(rng.integers(0, max(1, n - 40))), int(rng.integers(0, max(1, n - 40)))
            span = int(rng.integers(3, 40))
            tok[dst:dst + span] = tok[src:src + span].copy()
        window = [None, 0, 1, 2, 3, 5, 8, 17, 50][seed % 9]
        pm = [(5, 64), (2, 7), (1, 3), (3, 4), (1, 1), (4, 20)][seed % 6]
        orders = [(0, 1, 2, 3, 4), (2,), (0, 3), (1, 2, 4)][seed % 4]
        fb = doc_features(tok, ds, orders, window, *pm)
        fs = doc_features_stream(tok, ds, orders, window, *pm)
        assert fs.same_values(fb) and fs.n_fallback == 0 and fb.n_fallback == 0, seed
        if seed % 12 == 0:
            assert_matches_brute(fb, tok, ds)


def test_argument_checks():
    tok = np.array([1, 2, 3], dtype=np.int64)
    with pytest.raises(ValueError):
        doc_features(np.array([EDGE]), [0, 1])
    with pytest.raises(ValueError):
        doc_features(tok, [0, 2])
    with pytest.raises(ValueError):
        doc_features(tok, [0, 3], orders=(0, 5))
    with pytest.raises(ValueError):
        doc_features(tok, [0, 3], window=-1)
    with pytest.raises(ValueError):
        doc_features_stream(tok, [0, 3], ptr_min=6, ptr_max=5)


def test_ns_per_token_runs():
    tok, ds = random_stream(2, n_docs=3, max_len=300, vocab=50)
    med, per = ns_per_token(tok, ds, window=16, repeats=3, return_all=True)
    assert med > 0 and len(per) == 3
