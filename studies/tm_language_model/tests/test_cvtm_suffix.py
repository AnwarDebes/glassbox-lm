"""tmlm.cvtm.suffix: the suffix clauses of docs/11 section 11.2 reproduce Kneser-Ney exactly as a
sum of clause votes (docs/13 A3 H8.2), in float64 and within the rounding bound in Q16.

GPU cases (the suffix-link search on CuPy) run only when CuPy sees a device:

    scripts/gpu_run.sh shared -- .venv-tm/bin/python -m pytest tests/test_cvtm_suffix.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tmlm.cvtm.kn_gpu import fit_kn
from tmlm.cvtm.suffix import INT32_MIN, Q16, SuffixClauses, unigram_full
from tmlm.eval.baselines.ngram import KNModel


def _gpu_ok() -> bool:
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


GPU = _gpu_ok()
needs_gpu = pytest.mark.skipif(not GPU, reason="no CUDA device visible to CuPy")
DATA_ROOT = Path(__file__).resolve().parents[1] / "data"      # read only; conftest points TMLM_DATA_DIR elsewhere
needs_data = pytest.mark.skipif(
    not (DATA_ROOT / "wikitext103" / "bpe8000" / "validation" / "meta.json").exists(),
    reason="the V 8000 WikiText-103 store is not built (data/ is not part of a clean checkout)")
TOL = 1e-9


def markov_corpus(seed: int, vocab: int, n_docs: int, max_len: int, conc: float = 0.3):
    rng = np.random.default_rng(seed)
    cum = np.cumsum(rng.dirichlet(np.full(vocab, conc), size=vocab), axis=1)
    lengths = rng.integers(0, max_len + 1, size=n_docs)
    lengths[rng.random(n_docs) < 0.125] = 0
    toks = []
    for n in lengths:
        if n == 0:
            continue
        t = [int(rng.integers(vocab))]
        for x in rng.random(int(n) - 1):
            t.append(min(int(np.searchsorted(cum[t[-1]], x)), vocab - 1))
        toks.extend(t)
    return np.asarray(toks, dtype=np.uint16), np.concatenate([[0], np.cumsum(lengths)]).astype(np.uint32)


def queries(kn: KNModel, tok, ds, seed: int, hidden_rate: float):
    rng = np.random.default_rng(seed)
    hidden = (rng.random(tok.shape[0]) < hidden_rate).astype(np.uint8) if hidden_rate > 0 else None
    ctx, valid = kn.contexts(tok, ds, np.arange(tok.shape[0]), hidden)
    return ctx, valid, np.asarray(tok).astype(np.int64)


def assert_logprob_exact(sc: SuffixClauses, kn: KNModel, ctx, valid, w):
    lp = sc.logprob(ctx, w, valid)
    p = kn.prob(ctx, w, valid)
    zero = p == 0
    assert np.all(np.isneginf(lp[zero]))
    diff = np.abs(lp[~zero] - np.log(p[~zero]))
    assert diff.size == 0 or float(diff.max()) <= TOL, float(diff.max())
    lq = sc.logprob_q(ctx, w, valid)
    bound = sc.rounding_bound(ctx, w, valid)
    err = np.abs(lq[~zero] / Q16 - lp[~zero])
    assert err.size == 0 or bool(np.all(err <= bound[~zero] + 1e-12)), float((err - bound[~zero]).max())
    return float(diff.max()) if diff.size else 0.0


CASES = [
    # order, vocab, pad, discount_mode, discount, unk_interp, seed
    (1, 11, True, "modified", 0.75, True, 1), (2, 11, True, "modified", 0.75, True, 2),
    (3, 40, False, "modified", 0.75, True, 3), (4, 40, True, "single", 0.6, True, 4),
    (5, 60, True, "modified", 0.75, True, 5), (5, 60, False, "modified", 0.75, False, 6),
    (5, 25, True, "single", 0.1, False, 7), (4, 300, True, "modified", 0.75, True, 8),
]


@pytest.mark.parametrize("order,vocab,pad,mode,disc,unk,seed", CASES)
def test_logprob_equals_kn_on_random_corpora(order, vocab, pad, mode, disc, unk, seed):
    tok, ds = markov_corpus(seed, vocab, n_docs=50, max_len=250)
    kn = KNModel(order, vocab, discount_mode=mode, discount=disc, unk_interp=unk, pad=pad).fit(tok, ds)
    sc = SuffixClauses.from_kn(kn, backend="numpy")
    tok2, ds2 = markov_corpus(seed + 1000, vocab, n_docs=12, max_len=200)
    for t, d in ((tok, ds), (tok2, ds2)):
        for rate in (0.0, 0.2):
            ctx, valid, w = queries(kn, t, d, seed, rate)
            assert_logprob_exact(sc, kn, ctx, valid, w)
    # lower-order models of the same fit (raw counts at level `order`)
    for k in range(1, order):
        sk = SuffixClauses.from_kn(kn, order=k, backend="numpy")
        ctx, valid, w = queries(kn, tok2, ds2, seed, 0.1)
        m = ctx[:, ctx.shape[1] - (k - 1):] if k > 1 else ctx[:, :0]
        lp = sk.logprob(m, w, np.minimum(valid, k - 1))
        p = kn.prob(ctx, w, valid, order=k)
        nz = p > 0
        assert float(np.abs(lp[nz] - np.log(p[nz])).max()) <= TOL


@pytest.mark.parametrize("seed", [21, 22])
def test_dynamic_program_is_bit_identical_to_kn_prob(seed):
    tok, ds = markov_corpus(seed, 30, n_docs=40, max_len=250)
    kn = KNModel(5, 30).fit(tok, ds)
    sc = SuffixClauses.from_kn(kn, backend="numpy", keep_p=True)
    assert np.array_equal(sc.p1, unigram_full(kn, 5)) and np.array_equal(sc.p1, kn._unigram_full(5))
    # decode every top-level type into its context and target; its DP value is KNModel.prob of them
    seqs = [(t,) for t in range(kn.vocab + 1)]
    for k in range(2, 6):
        pre = (kn.keys[k] >> np.uint64(16)).astype(np.int64)
        w = (kn.keys[k] & np.uint64(0xFFFF)).astype(np.int64)
        seqs = [seqs[p] + (int(x),) for p, x in zip(pre, w)]
    fin = np.isfinite(sc.p[5])
    assert fin[kn.raw[5].cnt > 0].all()
    idx = np.flatnonzero(fin)
    ctx = np.array([seqs[i][:4] for i in idx], dtype=np.int64)
    w = np.array([seqs[i][4] for i in idx], dtype=np.int64)
    real = w < kn.vocab
    assert np.array_equal(sc.p[5][idx][real], kn.prob(ctx[real], w[real]))


def test_fired_matches_kn_context_ids_and_explain_lists_fired_levels():
    tok, ds = markov_corpus(31, 40, n_docs=40, max_len=250)
    kn = KNModel(5, 40).fit(tok, ds)
    sc = SuffixClauses.from_kn(kn, backend="numpy")
    tok2, ds2 = markov_corpus(1031, 40, n_docs=10, max_len=200)
    ctx, valid, w = queries(kn, tok2, ds2, 31, 0.2)
    fired = sc.fired(ctx, valid)
    cids = kn.context_ids(ctx, valid)
    for k in range(1, 5):
        lv = kn.level(k + 1, 5)
        cid = cids[k + 1]
        want = np.where((cid >= 0) & (lv.total[np.maximum(cid, 0)] > 0), cid, -1)
        assert np.array_equal(fired[:, k - 1], want)
    names = [f"t{i}" for i in range(40)]
    lp = sc.logprob(ctx, w, valid)
    for q in range(0, ctx.shape[0], 7):
        items, text = sc.explain(ctx[q], int(w[q]), token_text=names, valid=int(valid[q]))
        assert items[0]["clause"] == "bias"
        assert [it["k"] for it in items[1:]] == [k for k in range(1, 5) if fired[q, k - 1] >= 0]
        assert [it["context_id"] for it in items[1:]] == [int(fired[q, k - 1]) for k in range(1, 5) if fired[q, k - 1] >= 0]
        total = items[0]["vote"] + sum(it["d"] + it["vote"] for it in items[1:])
        assert abs(total - lp[q]) <= 1e-12
        assert text.count("\nIF tok[") == len(items) - 1
        for it in items[1:]:
            assert it["follower"] == (it["count"] > 0) and (it["vote"] > 0) <= it["follower"]
    items, text = sc.explain(ctx[5], int(w[5]), token_text=names)
    assert text.splitlines()[0].startswith("BIAS (always fires): ln p_1(")


def test_explain_rendering_format():
    tok = np.array([0, 1, 2, 0, 1, 2, 0, 1, 3], dtype=np.uint16)
    ds = np.array([0, 9], dtype=np.uint32)
    kn = KNModel(3, 4).fit(tok, ds)
    sc = SuffixClauses.from_kn(kn, backend="numpy")
    names = ["of", "the", "city", "sea"]
    items, text = sc.explain(np.array([0, 1]), 2, token_text=names)
    lines = text.splitlines()
    assert lines[1].startswith("IF tok[-1] = (the) THEN +") and "nats for city (continuation count" in lines[1]
    assert lines[2].startswith("IF tok[-2..-1] = (of, the) THEN +") and "nats for city (count 2); otherwise ln gamma = " in lines[2]
    assert items[2]["count"] == 2 and items[2]["stat"] == "raw"


def test_integer_form_and_state_hash():
    tok, ds = markov_corpus(41, 50, n_docs=40, max_len=250)
    kn = KNModel(5, 50).fit(tok, ds)
    a = SuffixClauses.from_kn(kn, backend="numpy")
    b = SuffixClauses.from_kn(KNModel(5, 50).fit(tok, ds), backend="numpy")
    assert a.state_hash() == b.state_hash()
    assert a.bq[a.bos] == INT32_MIN and np.isneginf(a.b[a.bos])
    for j in range(2, 6):
        assert a.dq[j].dtype == np.int32 and a.vq[j].dtype == np.int32
        assert np.array_equal(a.vq[j], np.rint(a.v[j] * Q16).astype(np.int32))
        assert np.array_equal(a.dq[j][~a.fires[j]], np.zeros(int((~a.fires[j]).sum()), np.int32))
    b.vq[5] = b.vq[5].copy()
    b.vq[5][np.flatnonzero(b.v[5])[0]] += 1
    assert a.state_hash() != b.state_hash()
    br = a.nbytes_breakdown()
    assert a.nbytes() == br["scoring"] == br["own"] + br["shared.keys"] > 0
    with pytest.raises(ValueError):
        a.logprob(np.zeros((1, 4), np.int64), np.array([50]))       # BOS is never a target


def test_zero_gamma_is_refused():
    tok, ds = markov_corpus(51, 20, n_docs=10, max_len=100)
    kn = KNModel(3, 20, discount_mode="single", discount=0.0).fit(tok, ds)
    with pytest.raises(ValueError, match="gamma = 0"):
        SuffixClauses.from_kn(kn, backend="numpy")


def test_pruned_models_are_exact_too():
    tok, ds = markov_corpus(61, 12, n_docs=40, max_len=300, conc=0.2)
    tok2, ds2 = markov_corpus(1061, 12, n_docs=8, max_len=200, conc=0.2)
    for mode, mass in (("recount", "renormalize"), ("full", "renormalize"), ("full", "backoff")):
        kn = fit_kn(tok, ds, 5, vocab=12, backend="numpy", min_counts={3: 2, 4: 2, 5: 3}, prune_cont=mode,
                    prune_mass=mass)
        sc = SuffixClauses.from_kn(kn, backend="numpy")
        ctx, valid, w = queries(kn, tok2, ds2, 61, 0.1)
        assert_logprob_exact(sc, kn, ctx, valid, w)


@needs_gpu
@pytest.mark.gpu
def test_cupy_links_equal_numpy_links():
    tok, ds = markov_corpus(71, 80, n_docs=60, max_len=300)
    kn = KNModel(5, 80).fit(tok, ds)
    a = SuffixClauses.from_kn(kn, backend="numpy")
    b = SuffixClauses.from_kn(kn, backend="cupy", chunk=1 << 10)
    assert a.state_hash() == b.state_hash()
    for j in range(2, 6):
        assert np.array_equal(a.v[j], b.v[j]) and np.array_equal(a.d[j], b.d[j])


@needs_data
def test_1m_slice_every_validation_position():
    from tmlm.data.corpus_store import load_split

    full = load_split("wikitext103", 8000, "train", root=DATA_ROOT)
    fds = np.asarray(full.doc_start).astype(np.int64)
    d = int(np.searchsorted(fds, 1_000_000, side="right")) - 1
    tok, ds = np.ascontiguousarray(full.tok[:int(fds[d])]), fds[:d + 1]
    kn = KNModel(5, 8000).fit(tok, ds)
    sc = SuffixClauses.from_kn(kn, backend="auto")
    val = load_split("wikitext103", 8000, "validation", root=DATA_ROOT)
    vtok = np.asarray(val.tok).astype(np.int64)
    vds = np.asarray(val.doc_start).astype(np.int64)
    ctx, valid = sc.contexts(vtok, vds, np.arange(vtok.shape[0]))
    kctx, kvalid = kn.contexts(vtok, vds, np.arange(vtok.shape[0]))
    assert np.array_equal(ctx, kctx) and np.array_equal(valid, kvalid)
    worst = assert_logprob_exact(sc, kn, ctx, valid, vtok)
    assert vtok.shape[0] == 282587 and worst <= TOL
    # the stream form agrees with KNModel.logprob (log2) on the same positions
    lp = sc.logprob_stream(vtok, vds)
    assert float(np.abs(lp / np.log(2.0) - kn.logprob(vtok, vds)).max()) <= TOL
