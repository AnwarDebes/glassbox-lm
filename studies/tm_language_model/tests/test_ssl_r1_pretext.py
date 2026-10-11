"""Tests of src/tmlm/ssl/pretext.py (docs/13 A7, R1 arms and the frequency-trap levers F1 to F4).

The central claim checked here: every data lever works through the engine's example set
(`CorpusArrays.targets`) with no engine change, and a per-epoch example set (F2) through one engine per
epoch that loads the previous state equals one engine trained with explicit, host-permuted batches."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from tmlm.data.synthetic import sym_freq_of, toy_corpus
from tmlm.engine import rng as R
from tmlm.engine.config import EngineConfig, Stream
from tmlm.engine.cpu.cpu_engine import CpuEngine
from tmlm.ssl import controls as K
from tmlm.ssl import pretext as P

ROOT = Path(__file__).resolve().parents[1]


def test_mix64_and_rng64_equal_the_reference():
    z = np.array([0, 1, 2, 3, 12345, 2 ** 40 + 7, 2 ** 63 + 5, 2 ** 64 - 1], dtype=np.uint64)
    assert [int(v) for v in P.mix64_np(z)] == [R.mix64(int(v)) for v in z]
    a = np.arange(50, dtype=np.uint64) * np.uint64(977)
    got = P.rng64_np(7, P.STREAM_F2, np.uint64(3), a, np.uint64(0))
    assert [int(v) for v in got] == [R.rng64(7, P.STREAM_F2, 3, int(x), 0) for x in a]


def test_arms_match_a7():
    assert sorted(P.ARMS) == ["A0", "A1", "A2", "A3", "A4", "A5", "A6"]
    base = EngineConfig(mode="masked", vocab=8000, n_clauses=65536, seed=100)
    a0 = P.engine_config("A0", 100)
    assert a0 == base
    a2 = P.engine_config("A2", 100)
    assert a2.neg_budget == 1 and a2.cand_tries == 4 and a2.mode == "masked"
    assert P.engine_config("A4", 100).mode == "causal" and P.engine_config("A5", 100).mode == "diffusion"
    assert P.ARMS["A1"].f1 and P.ARMS["A2"].f1 and P.ARMS["A6"].f1 and P.ARMS["A6"].random_kind_map
    assert P.ARMS["A3"].f2 and not P.ARMS["A3"].f1
    for name in ("A0", "A1", "A3", "A6"):
        assert P.engine_config(name, 1) == EngineConfig(mode="masked", vocab=8000, n_clauses=65536, seed=1)
    assert P.engine_config("A1", 1, n_clauses=4096).n_clauses == 4096
    assert (P.VOCAB, P.N_CLAUSES, P.N_TOKENS, P.EPOCHS, P.F1_T, P.F2_M) == (8000, 65536, 20_000_000, 4, 1e-4, 2000)


def test_f1_formula_and_draw():
    counts = np.array([0, 1, 10, 1000, 100000], np.int64)
    prob = P.f1_keep_prob(counts, t=1e-4)
    f = counts / counts.sum()
    exp = np.ones(5)
    exp[1:] = np.minimum(1.0, np.sqrt(1e-4 / f[1:]) + 1e-4 / f[1:])
    assert np.allclose(prob, exp)
    assert prob[0] == 1.0 and prob[1] == 1.0 and prob[4] < 0.05
    rng = np.random.default_rng(0)
    tok = rng.choice(5, size=20000, p=[0.001, 0.009, 0.09, 0.3, 0.6]).astype(np.uint16)
    kept = P.f1_targets(tok, 5, t=1e-3)
    assert np.all(np.diff(kept.astype(np.int64)) > 0)
    c = P.token_counts(tok, 5)
    always, thr = P.f1_keep_q32(c, 1e-3)
    for p in [0, 1, 17, 999, 19999]:  # the rule, position by position, with the reference mix64
        y = int(tok[p])
        keep = bool(always[y]) or (R.mix64(p) >> 32) < int(thr[y])
        assert keep == (p in set(kept.tolist()))
    # empirical keep rate per class close to the probability (20k positions)
    pr = P.f1_keep_prob(c, 1e-3)
    for y in (3, 4):
        rate = np.isin(np.flatnonzero(tok == y), kept).mean()
        assert abs(rate - pr[y]) < 0.03
    # seed and epoch free: the same subsample every call
    assert np.array_equal(kept, P.f1_targets(tok, 5, t=1e-3))


def test_f2_cap_per_class_and_epoch():
    rng = np.random.default_rng(1)
    tok = rng.choice(6, size=5000, p=[0.5, 0.2, 0.15, 0.1, 0.04, 0.01]).astype(np.uint16)
    m = 300
    t0 = P.f2_targets(tok, 0, m)
    t1 = P.f2_targets(tok, 1, m)
    for t in (t0, t1):
        cnt = np.bincount(tok[t], minlength=6)
        full = np.bincount(tok, minlength=6)
        assert np.all(cnt == np.minimum(full, m))
        assert np.all(np.diff(t.astype(np.int64)) > 0)
    assert not np.array_equal(t0, t1)          # redrawn per epoch
    assert np.array_equal(t0, P.f2_targets(tok, 0, m))
    # the kept positions of a capped class are those with the smallest rng64(0, STREAM_F2, e, p, 0)
    pos0 = np.flatnonzero(tok == 0)
    keys = [R.rng64(0, P.STREAM_F2, 0, int(p), 0) for p in pos0]
    exp = np.sort(pos0[np.argsort(keys, kind="stable")[:m]])
    assert np.array_equal(np.sort(t0[tok[t0] == 0].astype(np.int64)), exp)
    # composition with a base subset
    base = np.arange(0, 5000, 2, dtype=np.int64)
    tb = P.f2_targets(tok, 0, m, positions=base)
    assert set(tb.tolist()) <= set(base.tolist())


def test_random_kind_map_keeps_sizes_and_hierarchy():
    c = toy_corpus(vocab=60, n_docs=4, doc_len=100, seed=3)
    km, sf = P.random_kind_map(c.kind_map, c.sym_freq, 60)
    assert np.array_equal(km[0], np.arange(60))
    pi = P.random_permutation(60)
    assert sorted(pi.tolist()) == list(range(60))
    for k in (1, 2, 3):
        assert np.array_equal(km[k], c.kind_map[k][pi])
        assert sorted(np.bincount(km[k]).tolist()) == sorted(np.bincount(c.kind_map[k]).tolist())
    # sym_freq of the new map equals the docs/11 definition from the token pass
    assert np.array_equal(sf, sym_freq_of(c.tok, km, 60))
    # identity check: counts-based sym_freq reproduces the stored table of the original map
    assert np.array_equal(P.sym_freq_from_counts(c.kind_map, c.sym_freq[:60], 60), c.sym_freq)


def test_random_kind_map_on_the_real_store_tables():
    d = ROOT / "data" / "wikitext103" / "bpe8000" / "train"
    if not (d / "kind_map.npy").exists():
        pytest.skip("corpus store not present")
    km0 = np.load(d / "kind_map.npy")
    sf0 = np.load(d / "sym_freq.npy")
    assert np.array_equal(P.sym_freq_from_counts(km0, sf0[:8000], 8000), sf0)
    km, sf = P.random_kind_map(km0, sf0, 8000)
    # hierarchy: tokens sharing a kind-2 symbol share their kind-1 symbol (as in the original map)
    pairs0 = {(int(a), int(b)) for a, b in zip(km0[2], km0[1])}
    pairs = {(int(a), int(b)) for a, b in zip(km[2], km[1])}
    assert pairs == pairs0
    assert np.mean(km[3] != km0[3]) > 0.9


def test_slice_prefix_equals_tiny_lm_slice():
    sys.path.insert(0, str(ROOT / "experiments"))
    from m3_tiny_lm import slice_train

    c = toy_corpus(vocab=40, n_docs=7, doc_len=33, seed=2)
    for n in (1, 33, 34, 100, 231, 10_000):
        a, b = P.slice_prefix(c, n), slice_train(c, n)
        for f in ("tok", "doc_start", "kind_map", "stop", "sym_freq", "targets"):
            assert np.array_equal(getattr(a, f), getattr(b, f)), (n, f)


CFG = dict(mode="masked", vocab=40, n_clauses=64, seed=1, batch=16, sweep_period=8, n_bag_k=2)


def _host_batches(targets: np.ndarray, seed: int, epoch: int, batch: int) -> list[np.ndarray]:
    n = targets.shape[0]
    order = np.array([R.perm(seed, Stream.PERM, epoch, x, n) for x in range(n)], np.int64)
    pos = targets[order].astype(np.uint64)
    return [pos[i:i + batch] for i in range(0, n, batch)]


def test_targets_subset_is_the_example_set_of_the_engine():
    c = toy_corpus(vocab=40, n_docs=3, doc_len=50, seed=7)
    ad = P.arm_data(P.Arm("t", "masked", f1=True), c, vocab=40, t=0.02)
    sub = ad.targets(0)
    assert 0 < sub.shape[0] < c.tok.shape[0]
    cfg = EngineConfig(**CFG)
    a = CpuEngine(cfg, corpus=ad.corpus(0))
    assert a.n_examples == sub.shape[0]
    tot = 0
    for b in range(a.n_batches()):
        tot += a.train_batch(0, b)["n_examples"]
    assert tot == sub.shape[0]
    # bit-exact against explicit batches of the host permutation of the subset
    e = CpuEngine(cfg, corpus=ad.corpus(0))
    for batch in _host_batches(sub.astype(np.int64), cfg.seed, 0, cfg.batch):
        e.train_explicit(0, batch)
    assert a.state_hash() == e.state_hash()
    # gate mass only on the classes of kept positions
    G = a.state()["G"]
    assert set(np.flatnonzero(G).tolist()) <= set(np.unique(c.tok[sub]).tolist())


def test_per_epoch_example_sets_by_engine_rebuild_equal_explicit_training(tmp_path):
    """F2 path of experiments/ssl_r1_pretext.py train_arm (one engine per epoch, state loaded) against one
    engine fed explicit host-permuted batches of each epoch's example set; C0y checkpoint position."""
    sys.path.insert(0, str(ROOT / "experiments"))
    import ssl_r1_pretext as S

    c = toy_corpus(vocab=40, n_docs=3, doc_len=60, seed=11)
    arm = P.Arm("t2", "masked", f2=True)
    ad = P.arm_data(arm, c, vocab=40)
    import tmlm.ssl.pretext as PP
    old = PP.F2_M
    PP.F2_M = 5  # tiny cap so that the sets differ per epoch
    try:
        t0, t1 = ad.targets(0), ad.targets(1)
        assert not np.array_equal(t0, t1)
        cfg = EngineConfig(**CFG)
        rec = S.Rec(smoke=True)
        hashes = S.train_arm(ad, cfg, 2, tmp_path, "cpu", rec, young_fraction=0.3, yield_gpu=False)
        e = CpuEngine(cfg, corpus=ad.corpus(0))
        for ep, t in ((0, t0), (1, t1)):
            for batch in _host_batches(t.astype(np.int64), cfg.seed, ep, cfg.batch):
                e.train_explicit(ep, batch)
            if ep == 0:
                assert e.state_hash() == hashes["epochs"][1]
        assert e.state_hash() == hashes["epochs"][2]
    finally:
        PP.F2_M = old
    from tmlm.engine.checkpoint import load_checkpoint
    _, st, meta = load_checkpoint(tmp_path / "checkpoint_c0y.npz")
    nb = (t0.shape[0] + cfg.batch - 1) // cfg.batch
    assert meta["position"] == {"epoch": 0, "batch_index": K.c0y_batches(t0.shape[0], cfg.batch, 0.3)}
    assert int(st["step"][0]) == int(np.ceil(0.3 * nb))
    assert hashes["c0y"] == meta["state_hash"]
