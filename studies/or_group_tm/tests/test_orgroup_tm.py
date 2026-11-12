"""Unit tests of src/orgroup_tm.py and src/metrics.py (plain script, no pytest).

Run from the study folder:
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 .venv/bin/python -I tests/test_orgroup_tm.py
Prints one line per test and 'ALL n TESTS PASSED', or stops with exit status 1 at the first failure.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from metrics import auc, group_stats, groups_to_include_bits, plain_acc, tuned_acc
from orgroup_tm import OrGroupTM, pack_csr, pack_pairs

F = 8
PASSED = []


def ex(pairs, sides=2):
    """ids, bounds for a list of examples, each a tuple of word lists (one per side)."""
    ids, bounds = [], np.zeros((len(pairs), sides + 1), np.int64)
    for e, p in enumerate(pairs):
        bounds[e, 0] = len(ids)
        for s in range(sides):
            ids += list(p[s])
            bounds[e, s + 1] = len(ids)
    return np.array(ids, np.int32), bounds


def set_members(tm, c, g, words, value=0):
    for w in words:
        if tm.state[c, g, w] < 0:
            tm.nmem[c, g] += 1
        tm.state[c, g, w] = value


def step(tm, pairs, y, **kw):
    ids, bounds = ex(pairs, tm.sides)
    tm.fit_pass(ids, bounds, np.array(y), order=np.arange(len(y)), force_gate=True, **kw)


def must_fail(fn, what):
    try:
        fn()
    except AssertionError:
        return
    raise SystemExit(f"FAIL: a check that must fail passed: {what}")


def test(fn):
    fn()
    PASSED.append(fn.__name__)
    print("ok  ", fn.__name__, flush=True)
    return fn


@test
def evaluation_groups_and_empty_group():
    tm = OrGroupTM(1, F, sides=2, K=1)
    set_members(tm, 0, 0, [1, 2])
    set_members(tm, 0, 1, [5])
    ids, b = ex([([1], [5]), ([3], [5]), ([2, 3], [4]), ([2, 3], [4, 5])])
    assert tm.votes(ids, b).tolist() == [1, 0, 0, 1]
    tm.state[0, 1, 5] = -1
    tm.nmem[0, 1] = 0
    assert tm.votes(ids, b).tolist() == [0, 0, 0, 0], "a group without members must be false"
    # the same wrong expectation must be caught
    must_fail(lambda: (_ for _ in ()).throw(AssertionError) if tm.votes(ids, b).tolist() != [1, 0, 0, 1] else None,
              "empty group counted as true")


@test
def seed_deals_each_present_word_to_one_group_of_its_side():
    tm = OrGroupTM(1, F, sides=2, K=2, seed=3)
    step(tm, [([0, 3, 4], [1, 2])], [1])
    mem = tm.state[0] >= 0
    assert mem[:2].sum(0).tolist() == [1, 0, 0, 1, 1, 0, 0, 0], "side A words: one group each"
    assert mem[2:].sum(0).tolist() == [0, 1, 1, 0, 0, 0, 0, 0], "side B words: one group each"
    assert (tm.state[0][mem] == tm.seed_state).all() and (tm.state[0][~mem] == tm.init_state).all()
    assert tm.nmem_consistent() and tm.event_counts()["seed"] == 1


@test
def seed_all_gives_every_empty_group_every_present_word():
    tm = OrGroupTM(1, F, sides=2, K=2, seed_all=True)
    step(tm, [([0, 3, 4], [1, 2])], [1])
    mem = tm.state[0] >= 0
    assert mem[0].tolist() == mem[1].tolist() == [True, False, False, True, True, False, False, False]
    assert mem[2].tolist() == mem[3].tolist() == [False, True, True, False, False, False, False, False]
    assert tm.nmem.tolist() == [[3, 3, 2, 2]] and tm.nmem_consistent()


@test
def blank_clause_is_not_seeded_by_a_negative_example():
    tm = OrGroupTM(3, F, sides=2, K=1)
    step(tm, [([0, 3], [1, 2])], [0])
    assert (tm.state == tm.init_state).all() and tm.nmem.sum() == 0


@test
def max_seed_limits_seeding_per_example():
    for limit, expect in ((0, 3), (1, 1), (2, 2)):
        tm = OrGroupTM(3, F, sides=2, K=1, max_seed=limit)
        step(tm, [([0, 3], [1, 2])], [1])
        seeded = int((tm.nmem.sum(1) > 0).sum())
        assert seeded == expect and tm.event_counts()["seed"] == expect, (limit, seeded)
    tm = OrGroupTM(3, F, sides=2, K=1, max_seed=1)
    step(tm, [([0, 3], [1, 2]), ([4], [5])], [1, 1])
    assert int((tm.nmem.sum(1) > 0).sum()) == 2, "the next example may seed the next blank clause"


@test
def recruit_near_miss_then_member():
    tm = OrGroupTM(1, F, sides=2, K=1)
    set_members(tm, 0, 0, [1])
    set_members(tm, 0, 1, [5])
    step(tm, [([1], [6])], [1])
    assert tm.state[0, 1, 6] == 0 and tm.nmem[0, 1] == 2, "one recruit event from the initial state -1 makes a member"
    assert tm.state[0, 0, 1] == 0, "the true group is untouched when the clause does not fire"
    assert tm.event_counts()["recruit"] == 1


@test
def repress_then_two_recruits_needed():
    tm = OrGroupTM(1, F, sides=2, K=1)
    set_members(tm, 0, 0, [1])
    set_members(tm, 0, 1, [5])
    step(tm, [([1], [7])], [0])
    assert tm.state[0, 1, 7] == -2 and tm.nmem[0, 1] == 1
    step(tm, [([1], [7])], [1])
    assert tm.state[0, 1, 7] == -1 and tm.nmem[0, 1] == 1
    step(tm, [([1], [7])], [1])
    assert tm.state[0, 1, 7] == 0 and tm.nmem[0, 1] == 2


@test
def reject_hits_present_members_of_one_group():
    tm = OrGroupTM(1, F, sides=2, K=1, r=2)
    set_members(tm, 0, 0, [1, 2])
    set_members(tm, 0, 1, [5])
    step(tm, [([1], [5])], [0])
    hit_a, hit_b = tm.state[0, 0, 1] == -2, tm.state[0, 1, 5] == -2
    assert hit_a != hit_b, "exactly one group is hit"
    assert tm.state[0, 0, 2] == 0, "an absent member is never hit"
    assert tm.nmem_consistent()
    tm2 = OrGroupTM(1, F, sides=2, K=1, r=2, reject_all=True)
    set_members(tm2, 0, 0, [1, 2])
    set_members(tm2, 0, 1, [5])
    step(tm2, [([1], [5])], [0])
    assert tm2.state[0, 0, 1] == -2 and tm2.state[0, 1, 5] == -2 and tm2.nmem.tolist() == [[1, 0]]


@test
def reinforce_present_members_only():
    tm = OrGroupTM(1, F, sides=2, K=1)
    set_members(tm, 0, 0, [1, 2])
    set_members(tm, 0, 1, [5])
    step(tm, [([1, 3], [5])], [1])
    assert tm.state[0, 0, 1] == 1 and tm.state[0, 1, 5] == 1 and tm.state[0, 0, 2] == 0
    assert tm.state[0, 0, 3] == tm.init_state, "a present non-member is not touched when the clause fires"


@test
def reinforce_critical_and_share():
    tm = OrGroupTM(1, F, sides=2, K=1, reinforce="critical")
    set_members(tm, 0, 0, [1, 2])
    set_members(tm, 0, 1, [5])
    step(tm, [([1, 2], [5])], [1])
    assert tm.state[0, 0, 1] == 0 and tm.state[0, 0, 2] == 0, "two present members: none is critical"
    assert tm.state[0, 1, 5] == 1, "the one present member of its group is critical"
    step(tm, [([1, 3], [5])], [1])
    assert tm.state[0, 0, 1] == 1 and tm.state[0, 0, 2] == 0
    tm = OrGroupTM(1, F, sides=2, K=1, reinforce="share", seed=4)
    set_members(tm, 0, 0, [1, 2, 3])
    set_members(tm, 0, 1, [5])
    for _ in range(30):
        step(tm, [([1, 2], [5])], [1])
    assert tm.state[0, 0, 1] + tm.state[0, 0, 2] == 30 and tm.state[0, 0, 3] == 0, "one present member per event"
    assert tm.state[0, 0, 1] > 3 and tm.state[0, 0, 2] > 3, "both present members get a share"
    assert tm.state[0, 1, 5] == 30


@test
def decay_hits_only_weaker_present_members():
    tm = OrGroupTM(1, F, sides=2, K=1, reinforce="critical", p_decay=1.0)
    set_members(tm, 0, 0, [1, 2, 3], value=0)
    tm.state[0, 0, 1] = 4
    set_members(tm, 0, 1, [5, 6], value=0)
    step(tm, [([1, 2], [5, 6])], [1])
    assert tm.state[0, 0, 2] == -1 and tm.nmem[0, 0] == 2, "a present member weaker than the strongest one decays"
    assert tm.state[0, 0, 1] == 4 and tm.state[0, 0, 3] == 0, "the strongest and the absent member stay"
    assert tm.state[0, 1, 5] == 0 and tm.state[0, 1, 6] == 0, "equal present members do not decay"
    assert tm.nmem_consistent()
    tm0 = OrGroupTM(1, F, sides=2, K=1, reinforce="critical")
    set_members(tm0, 0, 0, [1, 2], value=0)
    tm0.state[0, 0, 1] = 4
    set_members(tm0, 0, 1, [5])
    step(tm0, [([1, 2], [5])], [1])
    assert tm0.state[0, 0, 2] == 0, "no decay by default"


@test
def reject_weakest_firing_group():
    for _ in range(5):
        tm = OrGroupTM(1, F, sides=2, K=2, r=2, reject="weakest", seed=_ + 1)
        set_members(tm, 0, 0, [1], value=5)
        set_members(tm, 0, 1, [2, 3], value=0)
        set_members(tm, 0, 2, [5], value=4)
        set_members(tm, 0, 3, [6], value=9)
        tm.state[0, 1, 3] = 7  # group 1 fires through word 2 (state 0) only when word 3 is absent
        step(tm, [([1, 2], [5, 6])], [0])
        assert tm.state[0, 1, 2] == -2, "the group whose strongest present member is weakest is hit"
        assert tm.state[0, 0, 1] == 5 and tm.state[0, 2, 5] == 4 and tm.state[0, 3, 6] == 9 and tm.state[0, 1, 3] == 7
        assert tm.nmem_consistent()
    tm = OrGroupTM(1, F, sides=2, K=1, r=2, reject="weakest")
    set_members(tm, 0, 0, [1, 2], value=0)
    tm.state[0, 0, 2] = 6
    set_members(tm, 0, 1, [5], value=3)
    step(tm, [([1, 2], [5])], [0])
    assert tm.state[0, 1, 5] == 1 and tm.state[0, 0, 1] == 0, "strength of a group = its strongest present member"
    try:
        OrGroupTM(1, F, reject="x")
    except ValueError:
        pass
    else:
        raise AssertionError("bad reject mode accepted")


@test
def modes_designate_the_right_groups():
    def fresh(mode, K):
        tm = OrGroupTM(1, F, sides=2, K=K, mode=mode, seed=5)
        if K == 2:
            for g, w in enumerate([1, 2, 5, 6]):
                set_members(tm, 0, g, [w])
        else:
            set_members(tm, 0, 0, [1])
            set_members(tm, 0, 1, [5])
        return tm
    # K = 2: side A true, both groups of side B false
    tm = fresh("side", 2)
    step(tm, [([1, 2], [7])], [1])
    assert sorted([int(tm.state[0, 2, 7]), int(tm.state[0, 3, 7])]) == [-1, 0], "the word goes to one false group of side B"
    tm = fresh("near", 2)
    step(tm, [([1, 2], [7])], [1])
    assert tm.state[0, 2, 7] == -1 and tm.state[0, 3, 7] == -1, "near: two false groups, nothing happens"
    # K = 1: both sides false
    for mode, expect in (("all", (0, 0)), ("near", (-1, -1)), ("side", (-1, -1))):
        tm = fresh(mode, 1)
        step(tm, [([3], [7])], [1])
        assert (int(tm.state[0, 0, 3]), int(tm.state[0, 1, 7])) == expect, mode


@test
def rejected_word_is_not_seeded_again():
    tm = OrGroupTM(1, F, sides=2, K=1, r=2, reject_all=True)
    set_members(tm, 0, 0, [1])
    set_members(tm, 0, 1, [5])
    step(tm, [([1], [5])], [0])
    assert tm.nmem.sum() == 0
    step(tm, [([1, 3], [5])], [1])
    assert tm.state[0, 0, 1] == -2 and tm.state[0, 0, 3] == 0 and tm.state[0, 1, 5] == -2
    assert tm.nmem.tolist() == [[1, 0]]


@test
def one_text_side_works():
    tm = OrGroupTM(1, F, sides=1, K=2, mode="near")
    set_members(tm, 0, 0, [1])
    set_members(tm, 0, 1, [2])
    ids, b = ex([([1, 2],), ([1, 3],)], sides=1)
    assert tm.votes(ids, b).tolist() == [1, 0]
    tm.fit_pass(ids, b, np.array([1, 1]), order=np.array([1]), force_gate=True)
    assert tm.state[0, 1, 3] == 0 and tm.state[0, 1, 1] == 0, "near miss: the present non-members join the false group"


@test
def random_training_invariants():
    rng = np.random.default_rng(0)
    n, Fbig = 300, 30
    Xp = (rng.random((2 * n, Fbig)) < 0.15).astype(np.uint8)
    P = np.c_[np.arange(n), n + np.arange(n)]
    ids, b = pack_pairs(Xp, P)
    y = rng.integers(0, 2, n)

    def run(seed, labels):
        tm = OrGroupTM(20, Fbig, sides=2, K=2, T=4, mode="side", neg_frac=0.25, seed=seed)
        for _ in range(3):
            tm.fit_pass(ids, b, labels)
        return tm
    t1, t2, t3 = run(1, y), run(1, y), run(2, y)
    assert t1.nmem_consistent()
    assert (t1.state == t2.state).all(), "the same seed gives the same model"
    assert (t1.state != t3.state).any(), "another seed gives another model"
    assert t1.state.max() <= t1.smax and t1.state.min() >= t1.smin
    t0 = OrGroupTM(20, Fbig, sides=2, K=2, T=4, mode="all", seed=1)
    for _ in range(3):
        t0.fit_pass(ids, b, np.zeros(n, np.int64))
    assert t0.state.max() < 0, "no member without an example on which the clause should fire"
    must_fail(lambda: (_ for _ in ()).throw(AssertionError) if t1.state.max() >= 0 else None,
              "claim that a trained model has no member")


@test
def tied_sides_keep_both_copies_equal():
    tm = OrGroupTM(1, F, sides=2, K=1, tied=True)
    step(tm, [([0, 3], [1, 2])], [1])
    assert (tm.state[0, 0] >= 0).tolist() == [True, True, True, True, False, False, False, False], \
        "a blank tied clause is seeded with the words of both texts"
    assert tm.twins_consistent() and tm.nmem_consistent() and tm.nmem.tolist() == [[4, 4]]
    ids, b = ex([([0], [2]), ([0], [5]), ([4], [5])])
    assert tm.votes(ids, b).tolist() == [1, 0, 0]
    step(tm, [([0], [5])], [1])  # near miss: the second text has no member, word 5 joins the one group
    assert tm.state[0, 0, 5] == 0 and tm.state[0, 1, 5] == 0 and tm.twins_consistent()
    rng = np.random.default_rng(3)
    n, Fbig = 300, 30
    Xp = (rng.random((2 * n, Fbig)) < 0.15).astype(np.uint8)
    idsr, br = pack_pairs(Xp, np.c_[np.arange(n), n + np.arange(n)])
    yr = rng.integers(0, 2, n)
    for kw in ({"K": 2, "mode": "side", "reinforce": "critical", "reject": "weakest", "seed_all": True, "p_decay": 0.2},
               {"K": 1, "mode": "near", "reinforce": "share", "reject": "all"}):
        t2 = OrGroupTM(20, Fbig, sides=2, T=4, tied=True, seed=2, **kw)
        for _ in range(3):
            t2.fit_pass(idsr, br, yr)
        assert t2.twins_consistent() and t2.nmem_consistent() and (t2.state >= 0).any()
    try:
        OrGroupTM(1, F, sides=1, tied=True)
    except ValueError:
        pass
    else:
        raise AssertionError("tied with one side was accepted")


@test
def weights_follow_the_share_of_right_firings():
    rng = np.random.default_rng(5)
    for share, lo, hi in ((0.9, 46, 64), (0.5, 22, 42), (0.1, 0, 18)):
        tm = OrGroupTM(1, F, sides=2, K=1, wmax=64, seed=3)
        set_members(tm, 0, 0, [1], value=20)
        set_members(tm, 0, 1, [5], value=20)
        ids, b = ex([([1], [5])] * 3000)
        y = (rng.random(3000) < share).astype(np.int64)
        tm.T = 1e9  # the gate then gives p = 0.5 for both labels; rejects cannot remove the strong members here
        tm.r = 1
        tm.smax = 32000
        tm.state[:] = np.where(tm.state >= 0, 30000, tm.state)
        tm.fit_pass(ids, b, y, order=np.arange(3000))
        assert lo <= tm.wts[0] <= hi, (share, int(tm.wts[0]))
    tm = OrGroupTM(2, F, sides=2, K=1, wmax=8)
    set_members(tm, 0, 0, [1])
    set_members(tm, 0, 1, [5])
    tm.wts[:] = [6, 2]
    ids, b = ex([([1], [5]), ([2], [5])])
    sc = tm.scores(ids, b)
    assert sc["unit"].tolist() == [1.0, 0.0] and sc["weight"].tolist() == [6.0, 0.0]
    assert abs(sc["logodds"][0] - np.log(6.5 / 2.5)) < 1e-12 and sc["logodds"][1] == 0.0
    assert set(OrGroupTM(1, F).scores(ids, b)) == {"unit", "ratio"}, "no weight readings unless wmax > 0"


@test
def ratio_reading_counts_clauses_matched_by_either_text():
    tm = OrGroupTM(3, F, sides=2, K=1, tied=True)
    for c, words in enumerate(([1], [2, 3], [5])):
        set_members(tm, c, 0, words)
        set_members(tm, c, 1, words)
    ids, b = ex([([1, 2], [1, 3]), ([1], [5]), ([4], [6]), ([1, 2, 5], [1, 2, 5])])
    sc = tm.scores(ids, b)
    assert sc["unit"].tolist() == [2.0, 0.0, 0.0, 3.0]
    assert sc["ratio"].tolist() == [1.0, 0.0, 0.0, 1.0], "both / either: 2 of 2, 0 of 2, 0 of 0, 3 of 3"
    ids, b = ex([([1, 2], [3]), ([1, 5], [1])])
    assert tm.scores(ids, b)["ratio"].tolist() == [0.5, 0.5]
    one = OrGroupTM(1, F, sides=1, K=1)
    i1, b1 = ex([([1],)], sides=1)
    assert "ratio" not in one.scores(i1, b1)


@test
def several_sides_and_the_atleast_mode():
    tm = OrGroupTM(1, F, sides=4, K=1, mode="atleast", min_true=2)
    ids, b = ex([([1], [2], [3], [4])], sides=4)
    tm.fit_pass(ids, b, np.array([1]), order=np.arange(1), force_gate=True)
    assert [np.flatnonzero(tm.state[0, g] >= 0).tolist() for g in range(4)] == [[1], [2], [3], [4]], "seeded side by side"
    ids, b = ex([([1], [2], [5], [6]), ([1], [7], [7], [7]), ([1], [2], [3], [4])], sides=4)
    assert tm.votes(ids, b).tolist() == [0, 0, 1]
    tm.fit_pass(ids, b, np.array([1, 1, 1]), order=np.array([0]), force_gate=True)
    assert tm.state[0, 2, 5] == 0 and tm.state[0, 3, 6] == 0, "two groups true: both false groups recruit"
    tm.fit_pass(ids, b, np.array([1, 1, 1]), order=np.array([1]), force_gate=True)
    assert tm.state[0, 1, 7] == -1 and tm.state[0, 2, 7] == -1, "one group true is below min_true: nothing is recruited"
    near = OrGroupTM(1, F, sides=4, K=1, mode="near")
    ids1, b1 = ex([([1], [2], [3], [4])], sides=4)
    near.fit_pass(ids1, b1, np.array([1]), order=np.arange(1), force_gate=True)
    ids2, b2 = ex([([1], [2], [3], [6]), ([1], [2], [5], [7])], sides=4)
    near.fit_pass(ids2, b2, np.array([1, 1]), order=np.array([1]), force_gate=True)
    assert near.state[0, 2, 5] == -1, "near: two false groups, nothing"
    near.fit_pass(ids2, b2, np.array([1, 1]), order=np.array([0]), force_gate=True)
    assert near.state[0, 3, 6] == 0, "near: one false group recruits"


@test
def constructor_and_input_checks():
    for kw in ({"mode": "x"}, {"init_state": 0}, {"sides": 0}, {"r": 0}, {"T": 0}, {"min_true": 0}):
        try:
            OrGroupTM(1, F, **kw)
        except ValueError:
            continue
        raise AssertionError(f"no error for {kw}")
    tm = OrGroupTM(1, F)
    ids, b = ex([([1], [9])])
    try:
        tm.fit_pass(ids, b, np.array([1]))
    except ValueError:
        pass
    else:
        raise AssertionError("word id out of range was accepted")


@test
def packers_agree():
    rng = np.random.default_rng(1)
    Xa = (rng.random((6, F)) < 0.4).astype(np.uint8)
    Xb = (rng.random((6, F)) < 0.4).astype(np.uint8)
    ids1, b1 = pack_pairs(np.vstack([Xa, Xb]), np.c_[np.arange(6), 6 + np.arange(6)])

    def csr(X):
        ids = np.concatenate([np.flatnonzero(r) for r in X]).astype(np.uint16)
        off = np.concatenate([[0], np.cumsum(X.sum(1))]).astype(np.int64)
        return ids, off
    ids2, b2 = pack_csr([csr(Xa), csr(Xb)])
    assert (ids1 == ids2).all() and (b1 == b2).all()
    assert b1[0].tolist() == [0, int(Xa[0].sum()), int(Xa[0].sum() + Xb[0].sum())]


@test
def metric_functions():
    y = np.array([1, 1, 0, 0])
    assert auc(np.array([3, 2, 1, 0]), y) == 1.0 and auc(np.array([0, 1, 2, 3]), y) == 0.0
    assert abs(auc(np.array([1, 1, 1, 1]), y) - 0.5) < 1e-12
    assert tuned_acc(np.array([3, 2, 1, 0]), y) == 1.0 and tuned_acc(np.array([0, 0, 0, 0]), y) == 0.5
    assert plain_acc(np.array([2, 0, 0, 1]), y) == 0.5
    st = group_stats([np.array([0, 8, 16]), np.array([1, 2])], n_hidden=8, m=8)
    assert st["n"] == 2 and abs(st["purity"] - 0.75) < 1e-9 and abs(st["coverage"] - (3 / 8 + 1 / 8) / 2) < 1e-9
    inc, A = groups_to_include_bits([[np.array([0, 8, 16]), np.array([], int)]], n_groups=2, V=24)
    assert A == 3 and inc.shape == (2, 6, 24)
    assert inc[0, :3].any(1).all() and not inc[0, 3:].any(), "full group: no empty option; empty group: all empty"
    assert not inc[1].any(), "odd rows stay empty"


if __name__ == "__main__":
    print(f"ALL {len(PASSED)} TESTS PASSED")
