"""CPU-only tests for experiments/p0_common.py (Phase 0 helpers). No GPU, no CAIR import."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import p0_common as common  # noqa: E402


@pytest.mark.parametrize("n_classes", [2, 10, 100, 16000])
def test_task_always_contains_the_largest_class(n_classes):
    # The stock fit() sets number_of_outputs = max(Y) + 1 (docs/02), so K-1 must occur.
    for n_graphs in (1, 7, 200):
        task = common.make_two_node_task(n_graphs, n_classes, seed=3, noise=0.0)
        assert task["y"].dtype == np.uint32
        assert int(task["y"].max()) == n_classes - 1
        assert int(task["y"].min()) >= 0
        assert task["y"].shape == (n_graphs,)


def test_task_is_deterministic_and_seed_dependent():
    a = common.make_two_node_task(500, 100, seed=42)
    b = common.make_two_node_task(500, 100, seed=42)
    c = common.make_two_node_task(500, 100, seed=43)
    for key in ("x1", "x2", "y", "clean"):
        assert np.array_equal(a[key], b[key])
    assert not np.array_equal(a["y"], c["y"])


def test_clean_labels_are_a_function_of_the_symbol_pair():
    task = common.make_two_node_task(4000, 1000, seed=1, noise=0.25)
    pair = 2 * task["x1"] + task["x2"]
    for value in range(4):
        labels = np.unique(task["y"][task["clean"] & (pair == value)])
        assert labels.size == 1
    expected = ((pair + 1) * common.PAIR_HASH_MULTIPLIER) % 2**32 % 1000
    assert np.array_equal(task["y"][task["clean"]], expected[task["clean"]])
    assert 0.2 < 1.0 - task["clean"].mean() < 0.3


def test_every_class_occurs_with_enough_noisy_examples():
    task = common.make_two_node_task(2000, 100, seed=5, noise=0.25)
    assert np.unique(task["y"]).size == 100


def test_task_rejects_bad_arguments():
    with pytest.raises(ValueError):
        common.make_two_node_task(0, 10, seed=0)
    with pytest.raises(ValueError):
        common.make_two_node_task(10, 1, seed=0)
    with pytest.raises(ValueError):
        common.make_two_node_task(10, 10, seed=0, noise=1.5)


def test_one_hot():
    y = np.array([0, 3, 1], dtype=np.uint32)
    hot = common.one_hot(y, 4)
    assert hot.shape == (3, 4) and hot.dtype == np.uint32
    assert np.array_equal(hot.sum(axis=1), np.ones(3))
    assert np.array_equal(hot.argmax(axis=1), y)


def test_summarize():
    stats = common.summarize([3.0, 1.0, 2.0, 10.0])
    assert stats["median"] == 2.5 and stats["min"] == 1.0 and stats["max"] == 10.0
    assert stats["n"] == 4 and stats["values"] == [3.0, 1.0, 2.0, 10.0]
    with pytest.raises(ValueError):
        common.summarize([])


def test_choose_n_respects_bounds_and_label_cap():
    assert common.choose_n(0.001, 10.0, 200, 60000, 10, 2**31) == 10000
    assert common.choose_n(1e-6, 10.0, 200, 60000, 10, 2**31) == 60000
    assert common.choose_n(1.0, 10.0, 200, 60000, 10, 2**31) == 200
    # 16000 outputs, cap 64 MB: at most 1000 rows of int32 labels.
    assert common.choose_n(0.001, 10.0, 200, 60000, 16000, 64_000_000) == 1000


def test_loglog_slope():
    ks = [10, 100, 1000]
    assert common.loglog_slope(ks, [1000.0, 100.0, 10.0]) == pytest.approx(-1.0)
    assert common.loglog_slope(ks, [5.0, 5.0, 5.0]) == pytest.approx(0.0)


def test_group_kernel_times_matches_exact_names():
    groups = common.group_kernel_times({
        "evaluate": 1.0, "select_clause_updates": 2.0, "update": 3.0, "update_message": 4.0,
        "calculate_messages": 5.0, "calculate_messages_conditional(int*)": 6.0,
        "select_clause_node": 7.0, "[CUDA memcpy HtoD]": 8.0, "exchange_messages": 9.0,
    })
    assert groups == {"evaluate": 1.0, "select_clause_updates": 2.0, "update": 7.0,
                      "calculate_messages": 11.0, "other": 24.0}
    frac = common.fractions(groups)
    assert sum(frac.values()) == pytest.approx(1.0)
    class_loop = sum(frac[name] for name in common.CLASS_LOOP_GROUPS)
    assert class_loop == pytest.approx(10.0 / 45.0)
    with pytest.raises(ValueError):
        common.fractions({"other": 0.0})


def test_run_worker_rejects_unknown_spec_keys():
    with pytest.raises(ValueError, match="unknown spec keys"):
        common.run_worker({"no_such_key": 1})
