"""Tests of src/tmlm/ssl/hao_world.py (docs/13 A7, H-AO world of H17.5) and of the pure parts of
experiments/ssl_r2_hao.py (verdict rule, tmu include decoding). CPU only, deterministic."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from tmlm.ssl import hao_world as H

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))


def test_world_is_seeded_and_split_30_70():
    a, b, c = H.make_world(7), H.make_world(7), H.make_world(8)
    assert np.array_equal(a.groups, b.groups) and np.array_equal(a.train_combo, b.train_combo)
    assert not np.array_equal(a.groups, c.groups)
    assert a.groups.shape == (4, 6) and np.unique(a.groups).size == 24
    assert a.n_combos == 1296 and int(a.train_combo.sum()) == 389
    assert a.distractors.size == 2000 - 24
    for cid in (0, 5, 777, 1295):
        assert a.combo_id(a.combo_members(cid)) == cid


def test_bags_follow_every_rule_train_and_test():
    w = H.make_world(3)
    for allowed, stream in ((w.train_combo, 1), (~w.train_combo, 2)):
        bags = H.sample_bags(w, 600, allowed, stream)
        assert all(H.check_bags(w, bags, allowed).values()), H.check_bags(w, bags, allowed)
        assert bags.X.shape == (600, 2000) and bags.X.dtype == np.uint8
        assert set(np.unique(bags.k_extra)) <= {0, 1, 2}
        assert np.all(bags.n_group_words == 4 + bags.k_extra)
        assert np.all(np.diff(bags.indptr) == 24 + bags.n_group_words)


def test_test_positives_are_novel_combinations():
    w = H.make_world(11)
    test = H.sample_bags(w, 400, ~w.train_combo, 2)
    for i in np.flatnonzero(test.y == 1):
        assert not any(w.train_combo[c] for c in H.contained_combos(w, test.bag(i)))


def test_sampling_is_deterministic():
    w = H.make_world(5)
    a = H.sample_bags(w, 200, w.train_combo, 1)
    b = H.sample_bags(w, 200, w.train_combo, 1)
    c = H.sample_bags(w, 200, w.train_combo, 9)
    assert np.array_equal(a.X, b.X) and np.array_equal(a.y, b.y)
    assert not np.array_equal(a.X, c.X)


def test_carve_combos_partitions_the_training_combinations():
    w = H.make_world(2)
    fit, val = H.carve_combos(w, w.train_combo, 0.2, 5)
    assert not np.any(fit & val) and np.array_equal(fit | val, w.train_combo)
    assert int(val.sum()) == round(0.2 * 389)


def test_linear_counts_cannot_separate_classes():
    """The number of group words is identically distributed in both classes (paired construction)."""
    w = H.make_world(4)
    bags = H.sample_bags(w, 1000, w.train_combo, 1)
    gw = bags.X[:, w.groups.reshape(-1)].sum(axis=1)
    assert np.array_equal(np.sort(gw[bags.y == 1]), np.sort(gw[bags.y == 0]))


def test_or_group_recovery():
    w = H.make_world(1)
    g = [set(map(int, r)) for r in w.groups]
    assert H.or_group_recovery(g, w.groups)["mean_jaccard"] == pytest.approx(1.0)
    assert H.or_group_recovery(g[::-1], w.groups)["mean_jaccard"] == pytest.approx(1.0)
    assert H.or_group_recovery([], w.groups)["mean_jaccard"] == 0.0
    half = [set(list(s)[:3]) for s in g]
    assert H.or_group_recovery(half, w.groups)["mean_jaccard"] == pytest.approx(0.5)
    merged = [g[0] | g[1], g[2], g[3]]
    r = H.or_group_recovery(merged, w.groups)
    assert r["mean_jaccard"] == pytest.approx((0.5 + 0 + 1 + 1) / 4)


def _print_hierarchy_reference(inc, hierarchy, append_negated):
    """A line-by-line transcription of PyHierarchicalTsetlinMachineCUDA.tm.CommonTsetlinMachine.print_hierarchy
    (c85ca4e7, tm.py:553-616), reading ta_action from `inc`, printing into a list."""
    out = []
    depth = len(hierarchy)
    size_levels = [0] * (depth + 1)
    size_levels[depth] = 1
    for d in range(depth - 1):
        size_levels[depth - d - 1] = hierarchy[depth - d - 1][1] * size_levels[depth - d]
    L = inc.shape[2]
    for i in range(inc.shape[0]):
        out.append("\nCLAUSE #%d: " % (i))
        previous_index = np.ones((depth - 1), dtype=np.int32) * -1
        for j in range(size_levels[1]):
            component_remainder = j
            left, right, inside = [], [], []
            feature_base = 0
            size = hierarchy[0][1]
            for d in range(1, depth):
                depth_d_node_index = component_remainder % hierarchy[d][1]
                component_remainder = component_remainder // hierarchy[d][1]
                if (hierarchy[d][0] != H.OR_ALTERNATIVES) and (hierarchy[d][0] != H.AND_ALTERNATIVES):
                    feature_base += size * depth_d_node_index
                    size *= hierarchy[d][1]
                if previous_index[d - 1] == -1:
                    left.append("(")
                elif depth_d_node_index == 0 and previous_index[d - 1] != depth_d_node_index:
                    right.append(")")
                    left.insert(0, "(")
                elif previous_index[d - 1] != depth_d_node_index:
                    if hierarchy[d][0] in [H.AND_GROUP, H.AND_ALTERNATIVES]:
                        inside.append(" ∧ ")
                    elif hierarchy[d][0] in [H.OR_GROUP, H.OR_ALTERNATIVES]:
                        inside.append(" ∨ ")
                previous_index[d - 1] = depth_d_node_index
            out.extend(right + inside + left)
            lits = []
            for k in range(L):
                if inc[i, j, k]:
                    if (not append_negated) or k < L // 2:
                        lits.append("x%d" % (feature_base + k,))
                    else:
                        lits.append("¬x%d" % (feature_base + k - L // 2,))
            if len(lits) > 1:
                out.append("(" + " ∧ ".join(lits) + ")")
            elif len(lits) == 1:
                out.append(lits[0])
        out.append(")" * (depth - 1) + "\n")
    return "".join(out)


@pytest.mark.parametrize("neg", [True, False])
def test_render_parse_and_array_groups_agree(neg):
    rng = np.random.default_rng(0)
    F, A, K = 40, 3, 4
    hier = ((H.AND_GROUP, F), (H.OR_ALTERNATIVES, A), (H.AND_ALTERNATIVES, K))
    inc = rng.random((5, A * K, 2 * F if neg else F)) < 0.06
    inc[1] = False  # an empty clause
    text = H.render_hierarchy(inc, hier, neg)
    assert text == _print_hierarchy_reference(inc, hier, neg)
    parsed = H.parse_printed_concepts(text, K)
    assert parsed == H.concept_groups(inc, A, K, neg)
    assert parsed[1] == [set(), set(), set(), set()]


def test_render_matches_reference_on_the_noisy_parity_structure():
    rng = np.random.default_rng(1)
    hier = ((H.AND_GROUP, 3), (H.OR_ALTERNATIVES, 3), (H.AND_GROUP, 2), (H.OR_ALTERNATIVES, 3), (H.AND_GROUP, 2))
    inc = rng.random((3, 36, 6)) < 0.3
    assert H.render_hierarchy(inc, hier, True) == _print_hierarchy_reference(inc, hier, True)


def test_noisy_parity_matches_the_demo_rule():
    X, y_noisy, y = H.noisy_parity(20000, 1, 11, 0.2)
    assert X.shape == (20000, 12) and X.dtype == np.uint32
    rel = [0, 1, 3, 4, 6, 7, 9, 10]
    assert np.array_equal(y, X[:, rel].sum(axis=1) % 2)
    assert abs(float(np.mean(y_noisy != y)) - 0.2) < 0.01
    Xt, yt_noisy, yt = H.noisy_parity(1000, 1, 12, 0.0)
    assert np.array_equal(yt_noisy, yt)


def test_verdict_rule():
    E = pytest.importorskip("ssl_r2_hao")
    V = {"pairs": {"p": ["h", "f"]}, "hier_min": 0.9, "margin": 0.2}
    cells = [{"seed": s, "model": "h", "test_acc": 0.95} for s in (1, 2, 3)] + [{"seed": s, "model": "f", "test_acc": 0.7} for s in (1, 2, 3)]
    assert E.verdict_h17_5(cells, [1, 2, 3], V)["p"]["meets_H17_5"] is True
    cells[3]["test_acc"] = 0.8  # flat seed 1 closer: mean diff (0.15 + 0.25 + 0.25) / 3 > 0.2
    assert E.verdict_h17_5(cells, [1, 2, 3], V)["p"]["meets_H17_5"] is True
    cells[4]["test_acc"] = 0.8
    assert E.verdict_h17_5(cells, [1, 2, 3], V)["p"]["meets_H17_5"] is False
    assert E.verdict_h17_5(cells[:5], [1, 2, 3], V)["p"]["status"] == "incomplete"
    low = [{"seed": s, "model": "h", "test_acc": 0.85} for s in (1, 2, 3)] + [{"seed": s, "model": "f", "test_acc": 0.5} for s in (1, 2, 3)]
    assert E.verdict_h17_5(low, [1, 2, 3], V)["p"]["meets_H17_5"] is False


def test_tmu_include_decoding_matches_get_ta_action():
    E = pytest.importorskip("ssl_r2_hao")
    pytest.importorskip("tmu")
    w = H.make_world(6)
    tr = H.sample_bags(w, 200, w.train_combo, 1)
    for kind in ("tmu_tm", "tmu_ctm"):
        tm, _, _, _ = E.tmu_fit_predict(kind, {"clauses": 16, "T": 20, "s": 5.0}, 1, tr.X, tr.y, [], 1)
        cb = tm.clause_bank if kind == "tmu_ctm" else tm.clause_banks[1]
        inc = E.tmu_included(cb)
        assert inc.shape == (cb.number_of_clauses, cb.number_of_literals)
        for i in range(cb.number_of_clauses):
            for k in range(0, cb.number_of_literals, 97):
                assert bool(inc[i, k]) == bool(cb.get_ta_action(i, k))
        st = E.tmu_stats(tm, kind, w)
        assert 0 <= st["clauses_used"] <= st["clauses"] and 0.0 <= st["group_word_coverage"] <= 1.0
        tm2, _, _, _ = E.tmu_fit_predict(kind, {"clauses": 16, "T": 20, "s": 5.0}, 1, tr.X, tr.y, [], 1)
        assert E.tmu_state_sha(tm, kind) == E.tmu_state_sha(tm2, kind)
