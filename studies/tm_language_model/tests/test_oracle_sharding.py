"""docs/11 section 9 (clause sharding), experiment s5 on the oracle: the same model trained as 1, 2
and 4 shards with only the three exchanges of section 9 must hash equal to the unsharded oracle
after every batch, with equal DIAG counters, through seeding, recycling and several sweeps."""

from __future__ import annotations

import pytest

from tmlm.data.synthetic import toy_corpus
from tmlm.engine.config import EngineConfig
from tmlm.engine.oracle.oracle import OracleEngine
from tmlm.engine.oracle.shard import ShardedOracle


@pytest.mark.parametrize("n_shards", [2, 4])
@pytest.mark.parametrize("mode", ["masked", "causal"])
def test_sharded_oracle_hashes_equal_unsharded(mode, n_shards):
    corpus = toy_corpus(vocab=40, n_docs=4, doc_len=60, seed=7)
    cfg = EngineConfig(mode=mode, vocab=40, n_clauses=64, seed=3, batch=16, sweep_period=4, n_bag_k=2)
    one = OracleEngine(cfg, corpus=corpus)
    many = ShardedOracle(cfg, corpus, n_shards)
    for e in range(3):
        for b in range(one.n_batches()):
            d1 = one.train_batch(e, b)
            d2 = many.train_batch(e, b)
            assert d1 == d2, (e, b)
            assert one.state_hash() == many.state_hash(), (e, b)
            assert many.shards_agree_on_shared_state()
    assert sum(int(one.alive[j]) for j in range(cfg.n_clauses)) > 0  # something was learned
    assert many.exchange["ex1_entries"] > 0 and many.exchange["batches"] == 3 * one.n_batches()


def test_sharded_oracle_rejects_bad_shapes():
    corpus = toy_corpus(vocab=40, n_docs=2, doc_len=30, seed=1)
    with pytest.raises(ValueError):
        ShardedOracle(EngineConfig(mode="masked", vocab=40, n_clauses=30, seed=1, batch=8, n_bag_k=1), corpus, 4)
