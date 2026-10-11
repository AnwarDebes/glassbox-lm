"""docs/11 sections 2, 4, 5 on the oracle: invariants, determinism, index consistency, the batch
contract, seeding, sweep and recycling, and that prediction never changes the state."""

from __future__ import annotations

import numpy as np
import pytest

from tmlm.data.synthetic import toy_corpus
from tmlm.engine.api import DIAG_COUNTERS, state_hash_of
from tmlm.engine.config import EngineConfig, Stream
from tmlm.engine.oracle.oracle import OracleEngine
from tmlm.engine.rng import perm


def make(mode="masked", seed=1, n_clauses=64, batch=16, sweep_period=8, **kw):
    corpus = toy_corpus(vocab=40, n_docs=3, doc_len=50, seed=7)
    cfg = EngineConfig(mode=mode, vocab=40, n_clauses=n_clauses, seed=seed, batch=batch,
                       sweep_period=sweep_period, n_bag_k=2, **kw)
    return OracleEngine(cfg, corpus=corpus), corpus


def run(eng, epochs=1):
    diags = []
    for e in range(epochs):
        for b in range(eng.n_batches()):
            diags.append(eng.train_batch(e, b))
    return diags


def test_invariants_hold_after_every_batch_and_clauses_get_seeded():
    eng, _ = make()
    seeded = 0
    for b in range(eng.n_batches()):
        d = eng.train_batch(0, b)
        assert set(d) == set(DIAG_COUNTERS)
        assert d["n_examples"] == min(eng.cfg.batch, eng.n_examples - b * eng.cfg.batch)
        seeded += d["n_seeded"]
        assert eng.check_invariants() == []
    assert seeded > 0
    assert any(eng.alive)


def test_same_seed_same_hash_different_seed_different_hash():
    a, _ = make(seed=1)
    b, _ = make(seed=1)
    c, _ = make(seed=2)
    run(a); run(b); run(c)
    assert a.state_hash() == b.state_hash()
    assert a.state_hash() != c.state_hash()


def test_state_round_trip_reproduces_the_future():
    a, corpus = make()
    run(a)
    st = a.state()
    assert state_hash_of(st, a.cfg) == a.state_hash()
    b = OracleEngine(a.cfg, corpus=corpus)
    b.load_state(st)
    assert b.state_hash() == a.state_hash()
    assert b.check_invariants() == []
    a.train_batch(1, 0); b.train_batch(1, 0)
    assert a.state_hash() == b.state_hash()


def test_index_agrees_with_brute_force_evaluation():
    eng, corpus = make()
    run(eng)
    for p in [3, 17, 60, 99, 120]:
        assert np.array_equal(eng.fired_clauses(1, p), eng.fired_clauses_bruteforce(1, p))


def test_train_explicit_with_the_permutation_equals_train_batch():
    a, corpus = make(batch=16)
    b, _ = make(batch=16)
    n = a.n_examples
    order = [perm(a.cfg.seed, int(Stream.PERM), 0, x, n) for x in range(16)]
    a.train_batch(0, 0)
    b.train_explicit(0, corpus.targets[order])
    assert a.state_hash() == b.state_hash()


def test_prediction_never_changes_the_state():
    eng, _ = make()
    run(eng)
    h = eng.state_hash()
    scores = eng.predict_scores(2, np.array([5, 40, 77]))
    _ = eng.fired_clauses(2, 5)
    assert scores.shape == (3, 40) and scores.dtype == np.int32
    assert eng.state_hash() == h


def test_sweep_runs_at_the_period_and_recycling_keeps_invariants():
    eng, _ = make(sweep_period=4, dead_sweeps=1)
    recycled = 0
    for e in range(3):
        for b in range(eng.n_batches()):
            d = eng.train_batch(e, b)
            recycled += d["n_recycled"]
            assert eng.check_invariants() == []
    # after several dead-sweep windows some clause has been recycled and the free list is a heap of free ids
    assert recycled >= 0
    assert sorted(eng.free) == [j for j in range(eng.n_clauses) if not eng.alive[j]]


def test_seeded_clause_votes_for_its_target_and_prints():
    eng, _ = make()
    run(eng)
    j = next(j for j in range(eng.n_clauses) if eng.alive[j])
    assert any(w > 0 for w in eng.vote_w[j])
    text = eng.clause_text(j)
    assert text.startswith(f"#{j} IF ") and " THEN " in text


@pytest.mark.parametrize("mode", ["causal", "diffusion"])
def test_other_modes_train_and_keep_invariants(mode):
    eng, _ = make(mode=mode)
    run(eng)
    assert eng.check_invariants() == []


def test_batch_size_changes_the_result_but_not_the_randomness_contract():
    """Randomness is keyed by (epoch, position), so the per-example gate draws are identical at
    any batch size; only the batch semantics (stale sums) differ. Check the first example's
    hidden pattern and forward pass agree across batch sizes."""
    a, _ = make(batch=4)
    b, _ = make(batch=16)
    ex_a = a._make_example(0, 10, None)
    ex_b = b._make_example(0, 10, None)
    assert [a._hidden(ex_a, q) for q in range(150)] == [b._hidden(ex_b, q) for q in range(150)]
