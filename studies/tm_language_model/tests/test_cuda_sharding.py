"""docs/11 section 9 on the CUDA engine (experiment s5, M4 part; docs/13 H3.5): the same model trained as
2 and 4 `CudaEngine` shards on the one GPU with only the three exchanges of section 9 must hash equal to
the unsharded CudaEngine AND to the oracle after every batch, with equal DIAG counters, through seeding,
recycling and several sweeps; the shards' G and step must agree without exchange; predictions are the
summed partial scores. Skipped without CuPy or a GPU. Run under `scripts/gpu_run.sh shared`."""

from __future__ import annotations

import numpy as np
import pytest

from tmlm.data.synthetic import toy_corpus
from tmlm.engine.config import EngineConfig
from tmlm.engine.oracle.oracle import OracleEngine

cp = pytest.importorskip("cupy")
try:
    cp.cuda.Device(0).compute_capability
except Exception as exc:  # pragma: no cover
    pytest.skip(f"no CUDA device: {exc}", allow_module_level=True)

from tmlm.engine.cuda.cuda_engine import CudaEngine  # noqa: E402
from tmlm.engine.cuda.shard import ShardedCudaEngine  # noqa: E402


@pytest.mark.parametrize("n_shards", [2, 4])
@pytest.mark.parametrize("mode", ["masked", "causal"])
def test_sharded_cuda_hashes_equal_unsharded_cuda_and_oracle(mode, n_shards):
    corpus = toy_corpus(vocab=40, n_docs=4, doc_len=60, seed=7)
    cfg = EngineConfig(mode=mode, vocab=40, n_clauses=64, seed=3, batch=16, sweep_period=4, n_bag_k=2)
    oracle = OracleEngine(cfg, corpus=corpus)
    one = CudaEngine(cfg, corpus=corpus)
    many = ShardedCudaEngine(cfg, corpus, n_shards)
    for e in range(3):
        for b in range(one.n_batches()):
            d0 = oracle.train_batch(e, b)
            d1 = one.train_batch(e, b)
            d2 = many.train_batch(e, b)
            assert d0 == d1 == d2, (e, b, d0, d1, d2)
            h0, h1, h2 = oracle.state_hash(), one.state_hash(), many.state_hash()
            assert h0 == h1 == h2, (e, b)
            assert many.shards_agree_on_shared_state()
    assert int(one.state()["alive"].sum()) > 0
    assert many.exchange["ex1_entries"] > 0 and many.exchange["batches"] == 3 * one.n_batches()
    ids = np.arange(0, one.n_examples, 23)
    assert np.array_equal(one.predict_scores(3, ids), many.predict_scores(3, ids))
    assert np.array_equal(one.fired_clauses(3, 17), many.fired_clauses(3, 17))
    # foreign clauses are never alive on a shard, own free lists partition the free set
    alive = one.state()["alive"]
    for r, sh in enumerate(many.shards):
        a = sh.state()["alive"]
        lo, hi = r * many.per, (r + 1) * many.per
        assert a[:lo].sum() == 0 and a[hi:].sum() == 0
        assert np.array_equal(a[lo:hi], alive[lo:hi])
    assert many.n_free() == one.n_free()


def test_sharded_cuda_with_a_bigger_vocabulary_and_recycling():
    """V = 200, 2048 clauses, batch 64, sweep every 4 with dead_sweeps 1: the MIN_FREE top-up (exchange 3)
    and recycling run; compared with the unsharded CudaEngine only (the oracle would be slow here)."""
    corpus = toy_corpus(vocab=200, n_docs=8, doc_len=200, seed=3, n_stop=8)
    cfg = EngineConfig(mode="masked", vocab=200, n_clauses=2048, seed=1, batch=64, sweep_period=4, dead_sweeps=1, n_bag_k=2)
    one = CudaEngine(cfg, corpus=corpus)
    many = ShardedCudaEngine(cfg, corpus, 4)
    recycled = 0
    for e in range(2):
        for b in range(one.n_batches()):
            d1, d2 = one.train_batch(e, b), many.train_batch(e, b)
            assert d1 == d2, (e, b)
            assert one.state_hash() == many.state_hash(), (e, b)
            recycled += d1["n_recycled"]
    assert recycled > 0
    assert many.exchange["ex3_entries"] >= 0


def test_sharded_cuda_rejects_bad_shapes():
    corpus = toy_corpus(vocab=40, n_docs=2, doc_len=30, seed=1)
    with pytest.raises(ValueError):
        ShardedCudaEngine(EngineConfig(mode="masked", vocab=40, n_clauses=30, seed=1, batch=8, n_bag_k=1), corpus, 4)
