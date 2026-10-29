"""Positions (only the past is visible) and the 10 floors (normalised, picks, eps rule, candidates, determinism)."""

import numpy as np
import pytest
from helpers import random_corpus, synth_corpus

from astruct import floors
from astruct.data import CAL, FIT, HOLD
from astruct.positions import Positions


def test_history_bos_pad_and_due():
    c = random_corpus(n_rep=50, seed=2)
    r = int(np.flatnonzero([len(c.expected(i)) > 0 and len(c.tokens(i)) >= 4 for i in range(c.n)])[0])
    P = Positions(c, np.array([r]))
    tok = c.tokens(r)
    assert P.n == len(tok) and (P.x == tok).all()
    assert P.H[0, 0] == c.BOS and P.H[0, 1] == c.PAD and P.H[1, 0] == tok[0] and P.H[1, 1] == c.BOS
    assert P.H[3, 2] == tok[0]
    exp = sorted(map(int, c.expected(r)))
    assert P.due_n[0] == len(exp) and list(P.due_types(0)) == sorted(set(exp))
    # due placeholders drop by one after each emission of an expected placeholder
    for t in range(1, P.n):
        emitted = [int(x) for x in tok[:t] if c.is_ph[x]]
        left = list(exp)
        for x in emitted:
            if x in left:
                left.remove(x)
        assert P.due_n[t] == len(left) and list(P.due_types(t)) == sorted(set(left))
    # said_before: only earlier positions of the same reply count
    assert not P.said_before(np.array([0]), np.array([tok[0]]))[0]
    assert P.said_before(np.array([1]), np.array([tok[0]]))[0]


def test_positions_do_not_read_the_future():
    c = random_corpus(n_rep=80, seed=9)
    r = int(np.flatnonzero([len(c.tokens(i)) >= 6 for i in range(c.n)])[0])
    t = 3
    tok = c.tokens(r)
    alt = np.concatenate([tok[: t + 1], (tok[t + 1:-1] + 1) % (c.V - 1) + 1, [c.END]]) if len(tok) > t + 2 else tok
    c2 = c.append_replies([(r, alt)])
    P1, P2 = Positions(c, np.array([r])), Positions(c2, np.array([c.n]))
    for k in ("H", "R", "due_n", "S", "A", "t"):
        assert np.array_equal(getattr(P1, k)[: t + 1], getattr(P2, k)[: t + 1]), k


@pytest.fixture(scope="module")
def synth():
    c = synth_corpus("floors", n=240, seed=1)
    fs = floors.FloorSet(c, c.replies(FIT), picks={}, need=floors.CONFIGS)
    Pc, Ph = Positions(c, c.replies(CAL)), Positions(c, c.replies(HOLD))
    vc = fs.view(Pc)
    fs.fit_em(vc, "F2n3")
    fs.fit_em(vc, "F2n5")
    fs.picks.update({"F2": "F2n5", "F3": "F3b", "F4": "F4e2"})
    fs.fit_em(vc, "F6")
    return c, fs, Pc, Ph


def test_all_floors_normalised(synth):
    _c, fs, _Pc, Ph = synth
    vh = fs.view(Ph)
    idx = np.arange(Ph.n)
    memo = fs.vectors(vh, idx, list(floors.CONFIGS) + ["KN5"])
    for name in list(floors.CONFIGS) + ["KN5"]:
        P = memo[name]
        assert np.abs(P.sum(1) - 1).max() < 1e-9, name
        assert P.min() > 0, name
    # the conditional floors use the facts: on average they beat the unconditional KN-5 here
    bits = {n: -np.log2(memo[n][idx, Ph.x]).mean() for n in memo}
    assert bits["F1n6"] < bits["KN5"] and bits["F5"] < bits["KN5"]
    # EM weights: one row per frequency bucket, each a distribution
    for n in ("F2n3", "F2n5", "F6"):
        lam = fs.lam[n]
        assert lam.shape == (floors.N_FREQ, 2) and np.allclose(lam.sum(1), 1) and (lam >= 0).all()


def test_eps_rule(synth):
    c, fs, _Pc, Ph = synth
    vh = fs.view(Ph)
    idx = np.arange(min(Ph.n, 300))
    memo = fs.vectors(vh, idx, ["F3b", "F4e4"])
    P3, P4 = memo["F3b"], memo["F4e4"]
    for i, k in enumerate(idx):
        due = set(map(int, Ph.due_types(int(k))))
        pen = [p for p in c.ph_ids if p not in due] + ([c.END] if Ph.due_n[k] > 0 else [])
        keep = [w for w in range(c.V) if w not in pen]
        z = P3[i, keep].sum() + 1e-4 * P3[i, pen].sum()
        assert np.allclose(P4[i, keep], P3[i, keep] / z) and np.allclose(P4[i, pen], 1e-4 * P3[i, pen] / z)


def test_topk_ties_to_smaller_id():
    P = np.array([[0.1, 0.3, 0.3, 0.05, 0.25], [0.2, 0.2, 0.2, 0.2, 0.2]])
    ids, _vals = floors.topk(P, 3)
    assert ids[0].tolist() == [1, 2, 4] and ids[1].tolist() == [0, 1, 2]
    rng = np.random.default_rng(0)
    Q = rng.integers(0, 4, (50, 30)).astype(float)
    ids, _ = floors.topk(Q, 8)
    for r in range(50):
        ref = sorted(range(30), key=lambda v: (-Q[r, v], v))[:8]
        assert ids[r].tolist() == ref


def test_summary_candidates(synth):
    c, fs, _Pc, Ph = synth
    s = floors.summarize(fs, fs.view(Ph), "F6")
    off = s["cand_off"]
    for k in range(Ph.n):
        cl = s["cand"][off[k]:off[k + 1]].tolist()
        assert len(cl) == len(set(cl)) >= 8 and c.END in cl
        assert set(map(int, Ph.due_types(k))) <= set(cl)
        assert s["cand_rank"][off[k]:off[k] + 8].tolist() == list(range(8))
        p = s["cand_p"][off[k]:off[k + 1]]
        assert (np.diff(p[:8]) <= 0).all() and abs(p[0] - s["top1"][k]) == 0
    assert np.allclose(-np.log2(s["p_true"]), floors.bits_true(fs, fs.view(Ph), ["F6"])["F6"])


def test_floor_deterministic(synth):
    c, fs, Pc, Ph = synth
    fs2 = floors.FloorSet(c, c.replies(FIT), picks=dict(fs.picks), need=("F6",))
    vc2 = fs2.view(Pc)
    fs2.fit_em(vc2, "F2n5")
    fs2.fit_em(vc2, "F6")
    a = floors.summarize(fs, fs.view(Ph), "F6")
    b = floors.summarize(fs2, fs2.view(Ph), "F6")
    for k in ("cand", "cand_p", "p_true", "top1", "ent"):
        assert np.array_equal(a[k], b[k]), k
