"""docs/11 section 8b: the two knobs added on 2026-09-22 (seg_weighting, neg_budget) behave as
specified in the oracle and in the C++ engine, and leave the default behaviour unchanged."""

from __future__ import annotations

import numpy as np

from tmlm.data.synthetic import toy_corpus
from tmlm.engine.config import EngineConfig, derive
from tmlm.engine.cpu.cpu_engine import CpuEngine
from tmlm.engine.oracle.oracle import OracleEngine


def test_uniform_weighting_gives_equal_segment_weights():
    rec = derive(EngineConfig(mode="causal", vocab=40, n_clauses=8, n_bag_k=2, seg_weighting="recency"))
    uni = derive(EngineConfig(mode="causal", vocab=40, n_clauses=8, n_bag_k=2, seg_weighting="uniform"))
    assert rec.seg_ids == uni.seg_ids
    assert np.diff([0, *uni.seg_cum]).tolist() == [1] * len(uni.seg_ids)
    assert np.diff([0, *rec.seg_cum]).tolist()[:8] == [1, 2, 4, 8, 16, 32, 64, 128]


def test_neg_budget_caps_included_negated_literals_in_both_engines():
    corpus = toy_corpus(vocab=40, n_docs=3, doc_len=50, seed=7)
    for neg_budget in (0, 1):
        cfg = EngineConfig(mode="masked", vocab=40, n_clauses=64, seed=1, batch=16, sweep_period=8, n_bag_k=2,
                           p_neg=0.5, neg_budget=neg_budget)
        o, c = OracleEngine(cfg, corpus=corpus), CpuEngine(cfg, corpus=corpus)
        for e in range(3):
            for b in range(o.n_batches()):
                assert o.train_batch(e, b) == c.train_batch(e, b)
        assert o.state_hash() == c.state_hash()
        st = o.state()
        for j in range(cfg.n_clauses):
            n_neg = sum(1 for lit, s in zip(st["lit_id"][j], st["lit_state"][j]) if s >= 128 and lit != 0xFFFFFFFF and lit & 1)
            assert n_neg <= neg_budget
        assert o.check_invariants() == []


def test_defaults_keep_the_golden_hash_of_the_parity_configs():
    """The defaults are the tuned values of docs/11 section 7 (2026-09-22): an unchanged config
    gives the same hash as spelling them out."""
    corpus = toy_corpus(vocab=40, n_docs=3, doc_len=50, seed=7)
    a = OracleEngine(EngineConfig(mode="masked", vocab=40, n_clauses=32, seed=1, batch=16, n_bag_k=2), corpus=corpus)
    b = OracleEngine(EngineConfig(mode="masked", vocab=40, n_clauses=32, seed=1, batch=16, n_bag_k=2, seg_weighting="recency", neg_budget=2, s=128, p_ins=0.25, dead_sweeps=32, v_slots=64, k_cover=8, seed_cap=32, rho_seed=1.0, neg_mode="binomial"), corpus=corpus)
    for bb in range(a.n_batches()):
        a.train_batch(0, bb); b.train_batch(0, bb)
    assert a.state_hash() == b.state_hash()


def test_binomial_negatives_are_bit_exact_between_engines_and_distributed_as_binomial():
    """docs/11 N14: neg_mode = binomial. (1) oracle == C++ on hash and counters; (2) the number of
    drawn negatives per event has mean Q and matches Binomial(V-1, Q/(V-1)) on a chi-square test
    (the initial design's distributional-equivalence idea, in Track B form); (3) draws are distinct and != y."""
    import math
    from tmlm.engine.config import NEG_K_MAX
    corpus = toy_corpus(vocab=40, n_docs=3, doc_len=50, seed=7)
    cfg = EngineConfig(mode="masked", vocab=40, n_clauses=64, seed=1, batch=16, sweep_period=8, n_bag_k=2, neg_mode="binomial", q=1.0)
    o, c = OracleEngine(cfg, corpus=corpus), CpuEngine(cfg, corpus=corpus)
    for e in range(3):
        for b in range(o.n_batches()):
            assert o.train_batch(e, b) == c.train_batch(e, b)
    assert o.state_hash() == c.state_hash()
    # distribution of the selection count over many (a, j) pairs at V = 1000, Q = 0.5
    cfg2 = EngineConfig(mode="masked", vocab=1000, n_clauses=8, seed=3, n_bag_k=0, neg_mode="binomial", q=0.5)
    big = toy_corpus(vocab=1000, n_docs=2, doc_len=30, seed=1)
    o2 = OracleEngine(cfg2, corpus=big)
    counts = np.zeros(NEG_K_MAX + 1, dtype=np.int64)
    trials = 20000
    for a in range(trials):
        chosen = o2.draw_negative_set(a, 5, 17)
        assert len(set(chosen)) == len(chosen) and 17 not in chosen
        counts[len(chosen)] += 1
    n, p = 999, 0.5 / 999
    assert abs(counts @ np.arange(NEG_K_MAX + 1) / trials - 0.5) < 3 * math.sqrt(0.5 / trials)
    pmf = [math.comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(NEG_K_MAX + 1)]
    expected = np.array(pmf) * trials
    # pool bins with expected count < 5 into the last kept bin
    keep = int(np.argmax(expected < 5))
    obs = np.concatenate([counts[:keep], [counts[keep:].sum()]]).astype(float)
    exp = np.concatenate([expected[:keep], [expected[keep:].sum()]])
    chi2 = float(((obs - exp) ** 2 / exp).sum())
    dof = len(obs) - 1
    # p > 0.01 threshold: chi-square critical value at 0.01 for dof degrees (dof <= 6 here)
    critical = {1: 6.63, 2: 9.21, 3: 11.34, 4: 13.28, 5: 15.09, 6: 16.81}[dof]
    assert chi2 < critical, f"chi2 {chi2:.2f} > {critical} with {dof} dof: counts {counts[:keep + 1]}"


def test_cand_tries_keeps_the_rarest_candidate_and_stays_bit_exact():
    corpus = toy_corpus(vocab=40, n_docs=3, doc_len=50, seed=7)
    for tries in (1, 3):
        cfg = EngineConfig(mode="masked", vocab=40, n_clauses=64, seed=1, batch=16, sweep_period=8, n_bag_k=2, cand_tries=tries)
        o, c = OracleEngine(cfg, corpus=corpus), CpuEngine(cfg, corpus=corpus)
        for e in range(2):
            for b in range(o.n_batches()):
                assert o.train_batch(e, b) == c.train_batch(e, b)
        assert o.state_hash() == c.state_hash()
    # with 3 tries the chosen literal's symbol is at least as rare as the single draw's
    cfg1 = EngineConfig(mode="masked", vocab=40, n_clauses=8, seed=1, batch=16, n_bag_k=2, cand_tries=1)
    cfg3 = EngineConfig(mode="masked", vocab=40, n_clauses=8, seed=1, batch=16, n_bag_k=2, cand_tries=3)
    o1, o3 = OracleEngine(cfg1, corpus=corpus), OracleEngine(cfg3, corpus=corpus)
    from tmlm.engine.config import Stream, lit_unpack
    worse = 0
    for p in range(10, 100):
        ex1, ex3 = o1._make_example(0, p, None), o3._make_example(0, p, None)
        l1, l3 = o1._sample_candidate(ex1, Stream.SEED_PICK, 0, 0, True), o3._sample_candidate(ex3, Stream.SEED_PICK, 0, 0, True)
        if l1 is not None and l3 is not None and o3._freq(lit_unpack(l3)[1]) > o1._freq(lit_unpack(l1)[1]):
            worse += 1
    assert worse == 0


def test_binomial_cdf_handles_n_below_k_max():
    """Small vocabularies: n = V - 1 < NEG_K_MAX. The pmf is zero beyond n, the CDF is 1 from k = n
    on, and the table never raises (regression: math domain error when neg_mode = binomial became
    the default, 2026-09-22)."""
    from tmlm.engine.config import NEG_K_MAX, binomial_cdf_q32
    full = (1 << 32) - 1
    t = binomial_cdf_q32(3, 0.5, NEG_K_MAX)
    assert len(t) == NEG_K_MAX + 1
    assert t[0] == int(0.125 * 4294967296)
    assert all(v == full for v in t[3:])
    assert all(t[i] <= t[i + 1] for i in range(NEG_K_MAX))
    # every tiny masked config builds
    from tmlm.engine.config import EngineConfig, derive
    for V in (2, 3, 4, 8, 33):
        derive(EngineConfig(mode="masked", vocab=V, n_clauses=8, seed=1, n_bag_k=2))

