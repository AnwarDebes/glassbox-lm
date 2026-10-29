"""tmlm.cvtm.shard: sharded Kneser-Ney counting (context-hash partition, each shard counted on its
own, merged) is bit-identical to the unsharded `tmlm.cvtm.kn_gpu.fit_kn` and to `KNModel.fit`:
keys, counts, continuation counts, statistics, discounts, and the suffix-clause votes
(docs/11 section 11.6; docs/13 A3 H8.8 b).

GPU cases run only when CuPy sees a device (run under scripts/gpu_run.sh shared):

    scripts/gpu_run.sh shared -- .venv-tm/bin/python -m pytest tests/test_cvtm_shard.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tmlm.cvtm.kn_gpu import compare_models, fit_kn, model_hashes
from tmlm.cvtm.shard import (compare_suffix_clauses, empty_context_hash, fit_kn_sharded, mix64_array,
                             shard_of_hash, tuple_hash, type_hash, type_shards, unigram_type_hash)
from tmlm.cvtm.suffix import SuffixClauses
from tmlm.engine.rng import mix64
from tmlm.eval.baselines.ngram import KNModel


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
needs_gpu = pytest.mark.skipif(not GPU, reason="no CUDA device visible to CuPy")
DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
needs_data = pytest.mark.skipif(
    not (DATA_ROOT / "wikitext103" / "bpe8000" / "train" / "meta.json").exists(),
    reason="the V 8000 WikiText-103 store is not built (data/ is not part of a clean checkout)")


def markov_corpus(seed: int, vocab: int, n_docs: int, max_len: int, conc: float = 0.3):
    """Documents from a skewed first-order Markov chain (n-grams repeat); about 1 in 8 documents is
    empty and some are shorter than the order."""
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
        for x in rng.random(int(n) - 1):
            t.append(min(int(np.searchsorted(cum[t[-1]], x)), vocab - 1))
        toks.extend(t)
    doc_start = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    return np.asarray(toks, dtype=np.uint16), doc_start


# ------------------------------------------------------------------ hashing
def test_mix64_array_equals_the_scalar_reference():
    rng = np.random.default_rng(3)
    z = rng.integers(0, 2**63, size=200, dtype=np.uint64) * np.uint64(2) + np.uint64(1)
    z = np.concatenate([z, np.array([0, 1, 2**64 - 1], dtype=np.uint64)])
    got = mix64_array(np, z)
    assert [int(g) for g in got] == [mix64(int(x)) for x in z]


def test_tuple_hash_is_the_chain_of_type_hashes():
    th1 = unigram_type_hash(np, 20)
    assert int(th1[7]) == tuple_hash([7])
    th2 = type_hash(np, th1[[7]], [3])
    assert int(th2[0]) == tuple_hash([7, 3])
    assert tuple_hash([]) == empty_context_hash()
    assert tuple_hash([7, 3]) != tuple_hash([3, 7])


@pytest.mark.parametrize("S", [2, 4, 5])
def test_partition_is_a_function_of_the_context_content(S):
    """The same context tuple goes to the same shard in two different corpora (a content hash, not a
    corpus-dependent id)."""
    V = 12
    tok_a, ds_a = markov_corpus(1, V, 30, 80)
    tok_b, ds_b = markov_corpus(2, V, 30, 80)
    for tok, ds in ((tok_a, ds_a), (tok_b, ds_b)):
        kn = KNModel(4, V).fit(tok, ds)
        shards = type_shards(kn, S)
        # decode every level-3 type's context back to its token pair and compare with tuple_hash
        k2 = kn.keys[2]
        ctx3 = (kn.keys[3] >> np.uint64(16)).astype(np.int64)
        a = (k2[ctx3] >> np.uint64(16)).astype(np.int64)      # level-1 id = token
        b = (k2[ctx3] & np.uint64(0xFFFF)).astype(np.int64)
        want = np.array([tuple_hash([x, y]) % S for x, y in zip(a, b)], dtype=np.int8)
        assert np.array_equal(shards[3], want)


# ------------------------------------------------------------------ bit identity
@pytest.mark.parametrize("backend", BACKEND_PARAMS)
@pytest.mark.parametrize("S", [1, 2, 3, 4, 7])
@pytest.mark.parametrize("order,pad", [(2, True), (3, False), (5, True), (5, False)])
def test_sharded_fit_equals_unsharded(backend, S, order, pad):
    V = 30
    tok, ds = markov_corpus(10 + order, V, 60, 150)
    ref = fit_kn(tok, ds, order=order, vocab=V, backend="numpy", pad=pad)
    ref_cpu = KNModel(order, V, pad=pad).fit(tok, ds)
    sh = fit_kn_sharded(tok, ds, order=order, vocab=V, shards=S, backend=backend, pad=pad)
    assert compare_models(sh, ref) == []
    assert compare_models(sh, ref_cpu) == []
    assert model_hashes(sh) == model_hashes(ref)
    assert sh.fit_info["shards"] == S


@pytest.mark.parametrize("backend", BACKEND_PARAMS)
def test_sharded_suffix_votes_equal(backend):
    V = 40
    tok, ds = markov_corpus(7, V, 80, 200)
    ref = fit_kn(tok, ds, order=5, vocab=V, backend="numpy")
    sh = fit_kn_sharded(tok, ds, order=5, vocab=V, shards=4, backend=backend)
    sa = SuffixClauses.from_kn(ref, backend="numpy")
    sb = SuffixClauses.from_kn(sh, backend="numpy")
    assert compare_suffix_clauses(sa, sb) == []
    assert sa.state_hash() == sb.state_hash()


def test_compare_suffix_clauses_finds_a_one_ulp_change():
    V = 25
    tok, ds = markov_corpus(8, V, 40, 120)
    kn = fit_kn(tok, ds, order=4, vocab=V, backend="numpy")
    sa = SuffixClauses.from_kn(kn, backend="numpy")
    sb = SuffixClauses.from_kn(kn, backend="numpy")
    assert compare_suffix_clauses(sa, sb) == []
    i = int(np.flatnonzero(sb.v[3] != 0)[0])
    sb.v[3] = sb.v[3].copy()
    sb.v[3][i] = np.nextafter(sb.v[3][i], np.inf)
    assert any(d.startswith("v[3]") for d in compare_suffix_clauses(sa, sb))


@pytest.mark.parametrize("S", [1, 4])
def test_single_discount_mode(S):
    V = 20
    tok, ds = markov_corpus(4, V, 30, 100)
    ref = fit_kn(tok, ds, order=3, vocab=V, backend="numpy", discount_mode="single", discount=0.6)
    sh = fit_kn_sharded(tok, ds, order=3, vocab=V, shards=S, backend="numpy", discount_mode="single", discount=0.6)
    assert compare_models(sh, ref) == []


@pytest.mark.parametrize("S", [1, 3])
def test_degenerate_corpora(S):
    V = 9
    cases = [
        (np.zeros(0, dtype=np.uint16), np.array([0], dtype=np.int64)),                  # no document
        (np.zeros(0, dtype=np.uint16), np.array([0, 0, 0], dtype=np.int64)),            # empty documents
        (np.array([3], dtype=np.uint16), np.array([0, 1], dtype=np.int64)),             # one token
        (np.array([1, 2, 1, 2, 1], dtype=np.uint16), np.array([0, 2, 2, 5], dtype=np.int64)),
    ]
    for tok, ds in cases:
        for order in (1, 2, 5):
            ref = fit_kn(tok, ds, order=order, vocab=V, backend="numpy")
            sh = fit_kn_sharded(tok, ds, order=order, vocab=V, shards=S, backend="numpy")
            assert compare_models(sh, ref) == []


def test_routing_matches_type_shards_and_every_shard_is_used():
    V = 60
    tok, ds = markov_corpus(21, V, 120, 200, conc=1.0)
    S = 4
    sh = fit_kn_sharded(tok, ds, order=5, vocab=V, shards=S, backend="numpy")
    ts = type_shards(sh, S)
    bal = sh.fit_info["balance"]
    for k in range(2, 6):
        counts = np.bincount(ts[k].astype(np.int64), minlength=S).tolist()
        assert bal[f"level{k}"]["types"] == counts
        assert sum(bal[f"level{k}"]["positions"]) == bal["level1"]["positions"]
        assert min(counts) > 0
    assert (ts[1] == empty_context_hash() % S).all()
    assert shard_of_hash(np, np.array([empty_context_hash()], dtype=np.uint64), S)[0] == empty_context_hash() % S


def test_argument_checks():
    tok, ds = markov_corpus(5, 10, 5, 20)
    with pytest.raises(NotImplementedError):
        fit_kn_sharded(tok, ds, order=3, vocab=10, backend="numpy", min_counts={3: 2})
    with pytest.raises(ValueError):
        fit_kn_sharded(tok, ds, order=3, vocab=10, shards=0, backend="numpy")
    with pytest.raises(ValueError):
        fit_kn_sharded(tok, ds, order=3, vocab=None, backend="numpy")


# ------------------------------------------------------------------ GPU and real data
@needs_gpu
@pytest.mark.gpu
def test_gpu_sharded_equals_gpu_unsharded_larger_corpus():
    V = 300
    tok, ds = markov_corpus(33, V, 400, 600, conc=0.1)
    ref = fit_kn(tok, ds, order=5, vocab=V, backend="cupy")
    for S in (2, 4):
        sh = fit_kn_sharded(tok, ds, order=5, vocab=V, shards=S, backend="cupy")
        assert compare_models(sh, ref) == []


@needs_gpu
@needs_data
@pytest.mark.gpu
def test_gpu_sharded_on_a_1m_training_slice():
    from tmlm.data.corpus_store import load_split

    full = load_split("wikitext103", 8000, "train", root=DATA_ROOT)
    fds = np.asarray(full.doc_start).astype(np.int64)
    d = int(np.searchsorted(fds, 1_000_000, side="right")) - 1
    tok = np.ascontiguousarray(full.tok[:int(fds[d])])
    ds = np.ascontiguousarray(fds[:d + 1])
    ref = fit_kn(tok, ds, order=5, vocab=8000, backend="cupy")
    sh = fit_kn_sharded(tok, ds, order=5, vocab=8000, shards=4, backend="cupy")
    assert compare_models(sh, ref) == []
    sa = SuffixClauses.from_kn(ref, backend="auto")
    sb = SuffixClauses.from_kn(sh, backend="auto")
    assert compare_suffix_clauses(sa, sb) == []
