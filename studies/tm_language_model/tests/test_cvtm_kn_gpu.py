"""tmlm.cvtm.kn_gpu: the GPU (and host) Kneser-Ney fit is array-identical to KNModel.fit, and the
count pruning of docs/13 A3 H8.9 has the documented semantics.

GPU cases run only when CuPy sees a device (run under scripts/gpu_run.sh shared):

    scripts/gpu_run.sh shared -- .venv-tm/bin/python -m pytest tests/test_cvtm_kn_gpu.py
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from tmlm.cvtm.kn_gpu import (compare_models, discounts_from_counts_of_counts, fit_kn, model_hashes,
                              model_nbytes)
from tmlm.eval.baselines.ngram import KNModel, modified_discounts


def _gpu_ok() -> bool:
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


GPU = _gpu_ok()
BACKENDS = ["numpy"] + (["cupy"] if GPU else [])
# every valid (prune_cont, prune_mass) pair; "backoff" needs prune_cont="full"
PRUNE_COMBOS = [("recount", "renormalize"), ("full", "renormalize"), ("full", "backoff")]
needs_gpu = pytest.mark.skipif(not GPU, reason="no CUDA device visible to CuPy")


# ------------------------------------------------------------------ corpora
def markov_corpus(seed: int, vocab: int, n_docs: int, max_len: int, conc: float = 0.3):
    """Documents from a skewed first-order Markov chain (so that n-grams repeat); about 1 in 8
    documents is empty and some are shorter than the order."""
    rng = np.random.default_rng(seed)
    trans = rng.dirichlet(np.full(vocab, conc), size=vocab)
    cum = np.cumsum(trans, axis=1)
    lengths = rng.integers(0, max_len + 1, size=n_docs)
    lengths[rng.random(n_docs) < 0.125] = 0
    toks = []
    for n in lengths:
        if n == 0:
            continue
        t = [int(rng.integers(vocab))]
        u = rng.random(int(n) - 1)
        for x in u:
            t.append(min(int(np.searchsorted(cum[t[-1]], x)), vocab - 1))
        toks.extend(t)
    doc_start = np.concatenate([[0], np.cumsum(lengths)]).astype(np.uint32)
    return np.asarray(toks, dtype=np.uint16), doc_start


DATA_ROOT = Path(__file__).resolve().parents[1] / "data"      # read only (memory-mapped); conftest points
needs_data = pytest.mark.skipif(                                 # TMLM_DATA_DIR elsewhere, so pass root=
    not (DATA_ROOT / "wikitext103" / "bpe8000" / "validation" / "meta.json").exists(),
    reason="the V 8000 WikiText-103 store is not built (data/ is not part of a clean checkout)")


def load_real(split: str):
    from tmlm.data.corpus_store import load_split

    assert split in ("train", "validation")                     # the test split is never loaded here
    return load_split("wikitext103", 8000, split, root=DATA_ROOT)


def slice_1m():
    full = load_real("train")
    ds = np.asarray(full.doc_start).astype(np.int64)
    d = int(np.searchsorted(ds, 1_000_000, side="right")) - 1
    end = int(ds[d])
    return np.ascontiguousarray(full.tok[:end]), np.ascontiguousarray(ds[:d + 1])


# ------------------------------------------------------------------ discounts
def test_discounts_from_counts_of_counts_bit_identical():
    rng = np.random.default_rng(7)
    cases = [np.array([1, 1, 2, 3, 4, 9]), np.array([2, 2, 3]), np.array([1, 1, 1]), np.array([3, 4, 4, 5]),
             np.array([0, 0, 1]), np.array([5, 6, 7]), np.array([1, 2, 2, 2, 2, 3, 4, 4, 4, 4, 4, 4, 4])]
    for _ in range(300):
        cases.append(rng.integers(0, 7, size=int(rng.integers(1, 60))))
    for c in cases:
        ref = modified_discounts(c)
        coc = [int((c == i).sum()) for i in (1, 2, 3, 4)]
        got = discounts_from_counts_of_counts(*coc)
        assert [x.hex() for x in got] == [float(x).hex() for x in ref], (c, ref, got)


# ------------------------------------------------------------------ exact identity with KNModel.fit
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("order,vocab,pad,mode,seed", [
    (1, 5, True, "modified", 1), (2, 7, True, "modified", 2), (3, 13, False, "modified", 3),
    (4, 3, True, "single", 4), (5, 50, True, "modified", 5), (5, 50, False, "modified", 6),
    (4, 200, True, "modified", 7), (3, 31, True, "single", 8),
])
def test_fit_identical_to_reference(backend, order, vocab, pad, mode, seed):
    tok, ds = markov_corpus(seed, vocab, n_docs=40, max_len=160)
    ref = KNModel(order, vocab, discount_mode=mode, discount=0.4, pad=pad).fit(tok, ds)
    got = fit_kn(tok, ds, order, vocab=vocab, backend=backend, discount_mode=mode, discount=0.4, pad=pad)
    assert compare_models(ref, got) == []
    assert model_hashes(ref) == model_hashes(got)
    assert set(got.fit_info["phases_s"]) >= {"stream", "level1", "cont_counts", "level_stats"}
    assert got.fit_info["backend"] == backend and got.fit_info["pruning"] is None


@pytest.mark.parametrize("backend", BACKENDS)
def test_fit_edge_corpora(backend):
    # one empty corpus with only empty documents, a single one-token document, documents at the end empty
    for tok, ds in [(np.zeros(0, np.uint16), np.array([0, 0, 0], np.uint32)),
                    (np.array([2], np.uint16), np.array([0, 1], np.uint32)),
                    (np.array([1, 2, 1, 2, 1], np.uint16), np.array([0, 2, 5, 5, 5], np.uint32))]:
        for order in (1, 2, 5):
            ref = KNModel(order, 4).fit(tok, ds)
            got = fit_kn(tok, ds, order, vocab=4, backend=backend)
            assert compare_models(ref, got) == [], (tok, ds, order)


def test_fit_rejects_bad_input():
    tok, ds = markov_corpus(1, 10, 5, 20)
    with pytest.raises(ValueError):
        fit_kn(tok, ds, 3, vocab=None, backend="numpy")
    with pytest.raises(ValueError):
        fit_kn(tok, ds, 3, vocab=5, backend="numpy")                  # token ids >= vocab
    with pytest.raises(ValueError):
        fit_kn(tok, ds[:-1], 3, vocab=10, backend="numpy")            # doc_start does not end at len(tok)
    with pytest.raises(ValueError):
        fit_kn(tok, ds, 3, vocab=10, backend="torch")


@needs_data
def test_fit_1m_slice_identical_to_reference():
    tok, ds = slice_1m()
    ref = KNModel(5, 8000).fit(tok, ds)
    ref_h = model_hashes(ref)
    for backend in BACKENDS:
        got = fit_kn(tok, ds, 5, vocab=8000, backend=backend)
        assert compare_models(ref, got) == [], backend
        assert model_hashes(got) == ref_h
    assert ref.n_train_tokens == tok.shape[0]


# ------------------------------------------------------------------ pruning: an independent reference
def naive_kn(tok, doc_start, order, vocab, pad, m, mode):
    """Dictionary implementation of the pruning semantics of the kn_gpu docstring (tuples of token
    ids, no key encoding). Returns prob(ctx_tuple, w) and the kept type sets per level."""
    n = order
    stream, real = KNModel(order, vocab, pad=pad)._padded_stream(tok, doc_start)
    stream = [int(x) for x in stream]
    grams: list[dict] = [dict() for _ in range(n + 1)]
    for k in range(1, n + 1):
        for p in range(k - 1, len(stream)):
            t = tuple(stream[p - k + 1:p + 1])
            grams[k][t] = grams[k].get(t, 0) + int(bool(real[p]))
    for t in range(vocab + 1):
        grams[1].setdefault((t,), 0)
    kept: list = [None] * (n + 1)
    for k in range(n, 0, -1):
        if m.get(k, 0) <= 0:
            kept[k] = set(grams[k])
        else:
            s = {t for t, c in grams[k].items() if c >= m[k]}
            s |= {t[:-1] for t in kept[k + 1]} if k < n else set()
            kept[k] = s
    full: list = [None] * (n + 1)
    full[n] = dict(grams[n])
    for k in range(n - 1, 0, -1):
        c = Counter(s[1:] for s, v in full[k + 1].items() if v > 0)
        full[k] = {t: c.get(t, 0) for t in grams[k]}
    cnt: list = [None] * (n + 1)
    cnt[n] = {t: grams[n][t] for t in kept[n]}
    for k in range(n - 1, 0, -1):
        if mode == "full":
            cnt[k] = {t: full[k][t] for t in kept[k]}
        else:
            cnt[k] = {t: 0 for t in kept[k]}
            for s, v in cnt[k + 1].items():
                if v > 0:
                    assert s[1:] in cnt[k]
                    cnt[k][s[1:]] += 1
    disc: list = [None] * (n + 1)
    stats: list = [None] * (n + 1)
    for k in range(1, n + 1):
        vals = np.array([v for v in cnt[k].values() if v > 0], dtype=np.int64)
        disc[k] = modified_discounts(vals) if vals.size else (1.0, 2.0, 3.0)
        st: dict = {}
        for t, v in cnt[k].items():
            e = st.setdefault(t[:-1], [0, 0, 0, 0])
            e[0] += v
            e[1] += v == 1
            e[2] += v == 2
            e[3] += v >= 3
        stats[k] = st

    def dsc(k, c):
        d1, d2, d3 = disc[k]
        return d3 if c >= 3 else d2 if c == 2 else d1 if c == 1 else 0.0

    def prob(ctx: tuple, w: int) -> float:
        st1 = stats[1][()]
        tot1 = st1[0]
        c1 = cnt[1].get((w,), 0)
        d1, d2, d3 = disc[1]
        if tot1 > 0:
            p = max(c1 - dsc(1, c1), 0.0) / tot1 + (d1 * st1[1] + d2 * st1[2] + d3 * st1[3]) / tot1 / vocab
        else:
            p = 1.0 / vocab
        if w == vocab:
            p = 0.0
        for k in range(2, n + 1):
            if k - 1 > len(ctx):
                continue                     # context cut (hidden token): the level does not fire
            u = tuple(ctx[len(ctx) - (k - 1):])
            if u not in kept[k - 1] or u not in stats[k] or stats[k][u][0] == 0:
                continue
            tot, a1, a2, a3 = stats[k][u]
            c = cnt[k].get(u + (w,), 0)
            d1, d2, d3 = disc[k]
            p = max(c - dsc(k, c), 0.0) / tot + (d1 * a1 + d2 * a2 + d3 * a3) / tot * p
        return p

    return prob, kept


def _decode(kn: KNModel) -> list:
    seqs: list = [None] * (kn.order + 1)
    seqs[1] = [(t,) for t in range(kn.vocab + 1)]
    for k in range(2, kn.order + 1):
        pre = (kn.keys[k] >> np.uint64(16)).astype(np.int64)
        w = (kn.keys[k] & np.uint64(0xFFFF)).astype(np.int64)
        seqs[k] = [seqs[k - 1][p] + (int(x),) for p, x in zip(pre, w)]
    return seqs


def _queries(kn: KNModel, tok, ds, seed):
    rng = np.random.default_rng(seed)
    pos = np.arange(tok.shape[0])
    hidden = (rng.random(tok.shape[0]) < 0.15).astype(np.uint8)
    ctx, valid = kn.contexts(tok, ds, pos, hidden)
    return ctx, valid, np.asarray(tok).astype(np.int64)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("mode", ["recount", "full"])
@pytest.mark.parametrize("order,m,pad,seed", [
    (4, {3: 2, 4: 2}, True, 11), (5, {3: 2, 4: 2, 5: 2}, True, 12), (5, {3: 3, 4: 3, 5: 3}, True, 13),
    (5, {4: 2, 5: 3}, False, 14), (3, {3: 2}, True, 15),
])
def test_prune_matches_naive_semantics(backend, mode, order, m, pad, seed):
    vocab = 9
    tok, ds = markov_corpus(seed, vocab, n_docs=30, max_len=200, conc=0.2)
    # naive_kn implements prune_mass="renormalize"; the default "backoff" has its own naive reference in
    # tests/test_cvtm_review.py
    kn = fit_kn(tok, ds, order, vocab=vocab, backend=backend, min_counts=m, prune_cont=mode, pad=pad,
                prune_mass="renormalize")
    prob, kept = naive_kn(tok, ds, order, vocab, pad, m, mode)
    # kept type sets of the pruned levels equal the reference; every kept type's context is present
    seqs = _decode(kn)
    for k in range(2, order + 1):
        keys = kn.keys[k]
        assert np.all(keys[1:] > keys[:-1])
        assert int((keys >> np.uint64(16)).max()) < kn.n_types[k - 1]
        if m.get(k, 0) > 0:
            assert set(seqs[k]) == kept[k], k
    # probabilities equal the reference on held-out text with cut contexts
    tok2, ds2 = markov_corpus(seed + 100, vocab, n_docs=6, max_len=120, conc=0.2)
    for t, d in ((tok, ds), (tok2, ds2)):
        ctx, valid, w = _queries(kn, t, d, seed)
        got = kn.prob(ctx, w, valid)
        # the naive prob reads the context from its last `valid` tokens, like KNModel.context_ids
        ref = np.array([prob(tuple(int(x) for x in c[len(c) - v:]), int(y)) for c, v, y in zip(ctx, valid, w)])
        assert np.allclose(got, ref, rtol=1e-12, atol=0.0), float(np.max(np.abs(got - ref)))
    # every row of dist sums to one
    ctx, valid, _ = _queries(kn, tok2, ds2, seed)
    rows = kn.dist(ctx[:300], valid[:300]).sum(axis=1)
    assert np.allclose(rows, 1.0, rtol=0, atol=1e-12)
    rep = kn.fit_info["pruning"]
    assert rep["prune_cont"] == mode and rep["n_types_pruned"] == list(kn.n_types)
    assert rep["prune_mass"] == "renormalize"


def test_naive_reference_equals_knmodel_unpruned():
    tok, ds = markov_corpus(21, 9, n_docs=25, max_len=150, conc=0.2)
    kn = KNModel(5, 9).fit(tok, ds)
    prob, _ = naive_kn(tok, ds, 5, 9, True, {}, "recount")
    ctx, valid, w = _queries(kn, tok, ds, 21)
    ref = np.array([prob(tuple(int(x) for x in c[len(c) - v:]), int(y)) for c, v, y in zip(ctx, valid, w)])
    assert np.allclose(kn.prob(ctx, w, valid), ref, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_prune_zero_thresholds_is_the_exact_fit(backend):
    tok, ds = markov_corpus(31, 20, n_docs=20, max_len=120)
    ref = KNModel(5, 20).fit(tok, ds)
    for m in ({}, {3: 0, 4: 0, 5: 0}):
        for mode, mass in PRUNE_COMBOS:
            got = fit_kn(tok, ds, 5, vocab=20, backend=backend, min_counts=m, prune_cont=mode, prune_mass=mass)
            assert compare_models(ref, got) == []


def test_prune_rejects_bad_thresholds():
    tok, ds = markov_corpus(1, 10, 5, 30)
    for m in ({2: 2}, {6: 2}, {3: 3, 4: 2}, {4: 2}, {3: -1}, {3: 1.5}):
        with pytest.raises(ValueError):
            fit_kn(tok, ds, 5, vocab=10, backend="numpy", min_counts=m)
    with pytest.raises(ValueError):
        fit_kn(tok, ds, 5, vocab=10, backend="numpy", min_counts={3: 2, 4: 2, 5: 2}, prune_cont="kenlm")
    with pytest.raises(ValueError):
        fit_kn(tok, ds, 5, vocab=10, backend="numpy", min_counts={3: 2, 4: 2, 5: 2}, prune_mass="kenlm")
    with pytest.raises(ValueError):                                   # backoff needs the unpruned continuation counts
        fit_kn(tok, ds, 5, vocab=10, backend="numpy", min_counts={3: 2, 4: 2, 5: 2}, prune_cont="recount")


@needs_gpu
def test_prune_cupy_equals_numpy():
    tok, ds = markov_corpus(41, 60, n_docs=60, max_len=300, conc=0.2)
    for mode, mass in PRUNE_COMBOS:
        a = fit_kn(tok, ds, 5, vocab=60, backend="numpy", min_counts={3: 2, 4: 2, 5: 3}, prune_cont=mode,
                   prune_mass=mass)
        b = fit_kn(tok, ds, 5, vocab=60, backend="cupy", min_counts={3: 2, 4: 2, 5: 3}, prune_cont=mode,
                   prune_mass=mass)
        assert compare_models(a, b) == []


@needs_data
def test_prune_1m_slice_rows_sum_to_one():
    tok, ds = slice_1m()
    val = load_real("validation")
    vtok = np.asarray(val.tok).astype(np.int64)
    vds = np.asarray(val.doc_start).astype(np.int64)
    pos = np.linspace(0, vtok.shape[0] - 1, 150).astype(np.int64)
    backend = BACKENDS[-1]
    full = fit_kn(tok, ds, 5, vocab=8000, backend=backend)
    b_full = model_nbytes(full)["total"]
    for m in ({3: 2, 4: 2, 5: 2}, {3: 3, 4: 3, 5: 3}):
        for mode, mass in PRUNE_COMBOS:
            kn = fit_kn(tok, ds, 5, vocab=8000, backend=backend, min_counts=m, prune_cont=mode, prune_mass=mass)
            ctx, valid = kn.contexts(vtok, vds, pos)
            rows = kn.dist(ctx, valid).sum(axis=1)
            assert np.allclose(rows, 1.0, rtol=0, atol=1e-12)
            assert model_nbytes(kn)["total"] < b_full
            assert kn.n_types[1] == full.n_types[1] and kn.n_types[2] == full.n_types[2]
            assert all(kn.n_types[k] < full.n_types[k] for k in (3, 4, 5))
