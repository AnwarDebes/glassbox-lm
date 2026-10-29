"""The chain engine: equal to tally's modified KN when it has no condition; normalised; exact lookups."""

import numpy as np
import pytest
from helpers import random_corpus

from astruct import chain as ch
from astruct import floors
from astruct.paths import add_study_paths
from astruct.positions import Positions

add_study_paths(tally=True)
from tally.floor.kn import FALLBACK, KN


def _sets(c):
    return Positions(c, c.replies(0)), Positions(c, np.concatenate([c.replies(1), c.replies(2)]))


@pytest.mark.parametrize("n", [2, 3, 5])
def test_plain_kn_chain_equals_tally(n):
    c = random_corpus(n_rep=500, seed=n)
    Pc, Ps = _sets(c)
    Ts, ex, raw = floors.chain_inputs(f"KNc{n}", Pc)
    lv = ch.fit_chain(Ts, ex, raw, Pc.x, c.V, "kn")
    Ts2, ex2, _ = floors.chain_inputs(f"KNc{n}", Ps)
    mine = ch.apply_chain(np.full((Ps.n, c.V), 1.0 / c.V), ch.lookup_all(lv, Ts2, ex2), ch.Packed(lv))
    kn = KN(Pc.stream, c.V, c.END, order=n, mode="sent", discount_fallback=FALLBACK)
    ref = kn.dists(Ps.stream, np.arange(Ps.n))
    assert np.abs(mine - ref).max() < 1e-12
    for j in range(n):  # level j of the chain is order n - j of tally
        assert np.allclose(lv[j].D[1:], kn.D[n - j][1:], atol=1e-12)
    # tally's fast path agrees with the vectors at the true tokens
    assert np.abs(np.log(mine[np.arange(Ps.n), Ps.x]) - kn.logprob(Ps.stream)).max() < 1e-10
    # and the wrapper used by the floors equals tally's dists
    assert np.abs(floors.KNVec(kn, Ps.stream).dists(np.arange(Ps.n)) - ref).max() == 0.0


@pytest.mark.parametrize("name", ["F1n4", "F2c3", "F3a", "F3b", "F5"])
def test_chains_normalised_and_positive(name):
    c = random_corpus(n_rep=400, seed=7)
    Pc, Ps = _sets(c)
    Ts, ex, raw = floors.chain_inputs(name, Pc)
    lv = ch.fit_chain(Ts, ex, raw, Pc.x, c.V, floors.chain_mode(name))
    Ts2, ex2, _ = floors.chain_inputs(name, Ps)
    base = np.random.default_rng(0).random((Ps.n, c.V)) + 0.01
    base /= base.sum(1, keepdims=True)
    out = ch.apply_chain(base.copy(), ch.lookup_all(lv, Ts2, ex2), ch.Packed(lv))
    assert np.abs(out.sum(1) - 1).max() < 1e-12 and out.min() > 0
    # the counted contexts change the base somewhere (the chain is not vacuous)
    assert np.abs(out - base).max() > 1e-3


def test_unseen_context_falls_through():
    c = random_corpus(n_rep=200, seed=3)
    Pc, _Ps = _sets(c)
    Ts, ex, raw = floors.chain_inputs("F3a", Pc)
    lv = ch.fit_chain(Ts, ex, raw, Pc.x, c.V, "abs")
    q = np.array([[123456789, 987654321]], np.int64)
    assert lv[0].lookup(q, np.array([True]))[0] == -1
    assert lv[0].lookup(Ts[0][:1], np.array([False]))[0] == -1  # a level that does not exist here
    assert lv[0].lookup(Ts[0][:1], np.array([True]))[0] >= 0


def test_hash_collisions_are_caught(monkeypatch):
    c = random_corpus(n_rep=200, seed=4)
    Pc, _ = _sets(c)
    Ts, ex, raw = floors.chain_inputs("F3a", Pc)
    monkeypatch.setattr(ch, "tuple_hash", lambda T: (np.asarray(T)[:, 0] % 2).astype(np.uint64))
    with pytest.raises(ValueError, match="collision"):
        ch.fit_chain(Ts, ex, raw, Pc.x, c.V, "abs")
    # a lookup whose hash matches but whose tuple differs is not found
    lvl = ch.Level(np.array([[1, 2]], np.int64), np.array([1], np.uint64), np.array([0.5]), np.array([0, 1]),
                   np.array([0]), np.array([0.5]), np.array([0.0, 0.5]), False)
    assert lvl.lookup(np.array([[1, 2]]), np.array([True]))[0] == 0
    assert lvl.lookup(np.array([[3, 2]]), np.array([True]))[0] == -1


def test_kn_continuation_counts_by_hand():
    """Bigram KN on replies 'a b', 'c b', 'a b': unigram b has 2 distinct left words (a, c)."""
    from astruct.data import Corpus, _csr
    vocab = ["<end>", "<unk>", "a", "b", "c"]
    toks = [[2, 3, 0], [4, 3, 0], [2, 3, 0]]
    off, flat = _csr(toks)
    n = 3
    z = np.zeros(n, np.int64)
    arr = {"r_dlg": np.arange(n, dtype=np.int32), "r_turn": z.astype(np.int16), "r_h": z.astype(np.int16) + 50,
           "r_split": z.astype(np.int8), "r_train": z.astype(bool), "tok_off": off, "tok": flat.astype(np.int32),
           "act_off": np.zeros(n + 1, np.int64), "act_ids": np.zeros(0, np.int16), "pair_off": np.zeros(n + 1, np.int64),
           "pair_ids": np.zeros(0, np.int16), "exp_off": np.zeros(n + 1, np.int64), "exp_ids": np.zeros(0, np.int32),
           "S_hash": z, "A_hash": z, "concept32": -np.ones(5, np.int16), "concept64": -np.ones(5, np.int16)}
    meta = {"vocab": vocab, "act_names": [], "pair_names": [], "dialogue_ids": ["x"] * 3,
            "lists": {"top32": [], "cue40": [], "pairs64": []}}
    c = Corpus(arr, meta)
    P = Positions(c, np.arange(n))
    Ts, ex, raw = floors.chain_inputs("KNc2", P)
    lv = ch.fit_chain(Ts, ex, raw, P.x, c.V, "kn")
    uni = lv[1]  # context () : continuation counts
    i0 = uni.lookup(np.zeros((1, 1), np.int64), np.array([True]))[0]
    w = uni.w[uni.start[i0]:uni.start[i0 + 1]]
    # a: left BOS (2 replies, 1 distinct) -> 1; b: left a, c -> 2; c: left BOS -> 1; <end>: left b -> 1
    assert dict(zip(w.tolist(), [0] * len(w))).keys() == {0, 2, 3, 4}
    T = 5.0  # adjusted counts 1 + 2 + 1 + 1
    D = uni.D
    num = dict(zip(w.tolist(), uni.num[uni.start[i0]:uni.start[i0 + 1]].tolist()))
    assert abs(num[3] - (2 - D[2]) / T) < 1e-12 and abs(num[2] - (1 - D[1]) / T) < 1e-12
