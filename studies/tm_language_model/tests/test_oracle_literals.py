"""docs/11 sections 1.3 and 1.4 on the oracle: contexts, literal truth, hiding, causal mode."""

from __future__ import annotations

import numpy as np

from tmlm.data.synthetic import corpus_from_tokens
from tmlm.engine.config import EngineConfig, bag_seg, lit_pack, offset_seg, sym_edge, sym_mask
from tmlm.engine.oracle.oracle import OracleEngine

V = 20


def corpus():
    # two documents of 12 tokens; token ids chosen so that every position is distinct
    tok = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 1, 2, 3, 4, 5], dtype=np.uint16)
    doc_start = np.array([0, 12, 24], dtype=np.uint32)
    return corpus_from_tokens(tok, doc_start, V, n_stop=1)  # token 1 is the single stop word


def engine(mode="causal", **kw):
    kw.setdefault("mask_rate", 0.0)
    cfg = EngineConfig(mode=mode, vocab=V, n_clauses=8, seed=1, n_bag_k=13, **kw)
    return OracleEngine(cfg, corpus=corpus())


def lit_true(eng, epoch, p, seg, sym, neg=0):
    ex = eng._make_example(epoch, p, None)
    return eng._lit_true(ex, lit_pack(seg, sym, neg))


def test_exact_literals_read_the_corpus_at_the_offset():
    eng = engine("causal")
    p = 6  # tokens before p: positions 0..5 hold 1..6
    for o in range(-6, 0):
        assert lit_true(eng, 0, p, offset_seg(o), 7 + o)  # tok[p + o] = p + o + 1
        assert not lit_true(eng, 0, p, offset_seg(o), 7 + o, neg=1)
    assert not lit_true(eng, 0, p, offset_seg(-1), 5)  # wrong token


def test_edge_symbol_outside_the_document_and_mask_at_the_target():
    eng = engine("masked")
    p = 13  # second document starts at 12: offset -2 is outside it
    assert lit_true(eng, 0, p, offset_seg(-2), sym_edge(V))
    assert not lit_true(eng, 0, p, offset_seg(-1), sym_edge(V))
    assert lit_true(eng, 0, p, offset_seg(-1), 13)  # tok[12] = 13
    # the target itself is hidden: a literal at offset 0 does not exist, so check ctx directly
    ex = eng._make_example(0, p, None)
    assert eng._ctx(ex, p, 0) == sym_mask(V)


def test_causal_mode_has_no_right_side_and_hides_nothing():
    eng = engine("causal")
    assert not lit_true(eng, 0, 5, offset_seg(+1), 7)  # right segments are not enabled (N11)
    assert lit_true(eng, 0, 5, offset_seg(+1), 7, neg=1)  # negation of a disabled literal is true
    ex = eng._make_example(0, 5, None)
    assert all(not eng._hidden(ex, q) for q in range(24))


def test_masked_mode_hiding_is_epoch_consistent_and_seeded():
    eng = engine("masked", mask_rate=0.5)
    ex_a = eng._make_example(3, 5, None)
    ex_b = eng._make_example(3, 9, None)
    hidden_a = [eng._hidden(ex_a, q) for q in range(24)]
    hidden_b = [eng._hidden(ex_b, q) for q in range(24)]
    assert hidden_a == hidden_b  # depends on (epoch, position) only
    assert any(hidden_a) and not all(hidden_a)
    ex_c = eng._make_example(4, 5, None)
    assert [eng._hidden(ex_c, q) for q in range(24)] != hidden_a
    # a frozen bitmap replaces the function
    bitmap = np.zeros(24, dtype=np.uint8)
    bitmap[4] = 1
    ex_d = eng._make_example(3, 5, bitmap)
    assert eng._hidden(ex_d, 4) and not eng._hidden(ex_d, 3)
    assert eng._ctx(ex_d, 4, 0) == sym_mask(V)


def test_bag_literals_scan_the_bucket_inside_the_document():
    eng = engine("causal")
    p = 23  # last position of document 2 (positions 12..23 hold tokens 13..19, 1, 2, 3, 4, 5)
    seg = bag_seg(0, 0)  # left bucket 0: distances 9..16, positions 14 down to 7, cut at 12
    assert lit_true(eng, 0, p, seg, 13)  # position 12, distance 11
    assert lit_true(eng, 0, p, seg, 14)  # position 13, distance 10
    assert lit_true(eng, 0, p, seg, 15)  # position 14, distance 9
    assert not lit_true(eng, 0, p, seg, 16)  # position 15 is at distance 8: exact range, not the bag
    assert not lit_true(eng, 0, p, seg, 19)  # distance 4
    assert not lit_true(eng, 0, p, seg, 1, neg=0)  # the stop word is at distance 4 anyway
    # negation of a false bag literal is true
    assert lit_true(eng, 0, p, seg, 16, neg=1)


def test_bag_literal_never_crosses_the_document_boundary():
    eng = engine("causal")
    p = 14  # document 2; bucket 0 (distance 9..16) would reach positions -2..5, all outside document 2
    for sym in (1, 2, 3, 4, 5, 6):
        assert not lit_true(eng, 0, p, bag_seg(0, 0), sym)


def test_cluster_symbols_follow_kind_map():
    eng = engine("causal")
    p = 6
    for kind in (1, 2, 3):
        sym = int(eng.kind_map[kind][6])  # token at offset -1 is 6
        assert lit_true(eng, 0, p, offset_seg(-1), sym)


def test_polarity_symbols_are_never_true():
    eng = engine("causal")
    assert not lit_true(eng, 0, 6, offset_seg(-1), sym_mask(V) + 2)  # polarity range: reserved (N11)
