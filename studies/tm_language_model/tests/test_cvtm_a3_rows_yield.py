"""Tests for experiments/cvtm_a3_rows_yield.py: the launcher replaces exactly the memory wait and the
scoring call of experiments/cvtm_a3_rows.py, keeps the memory rule (current + peak < 0.80 max, poll),
hands the GPU over at every poll and before scoring, and passes the scoring call through unchanged.
No GPU, no data, no lock files are touched (the lock call is replaced by a counter)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import cvtm_a3_rows as a3  # noqa: E402
import cvtm_a3_rows_yield as ly  # noqa: E402

GIB = 1024 ** 3


@pytest.fixture()
def patched(monkeypatch):
    calls = {"yield": 0, "orig_wait": [], "orig_score": 0, "sleep": []}
    monkeypatch.setattr(a3, "wait_for_memory", a3.wait_for_memory)
    monkeypatch.setattr(a3, "score", a3.score)
    monkeypatch.setattr(ly, "YIELDS", [])
    monkeypatch.setattr(a3, "METRICS", {})

    def fake_yield():
        calls["yield"] += 1
        return calls["yield"] == 2            # one hand-over, at the second call

    monkeypatch.setattr(ly.locks, "yield_gpu_if_requested", fake_yield)
    monkeypatch.setattr(ly, "_ORIG_WAIT", lambda peak, tag: calls["orig_wait"].append((peak, tag)))

    def fake_score(core, ev, lnp, ks, feats=None):
        calls["orig_score"] += 1
        return {"core": core, "ev": ev, "lnp": lnp, "ks": ks, "feats": feats}

    monkeypatch.setattr(ly, "_ORIG_SCORE", fake_score)
    monkeypatch.setattr(ly.time, "sleep", lambda s: calls["sleep"].append(s))
    return calls


def test_install_replaces_exactly_the_two_globals(patched):
    other = {k: getattr(a3, k) for k in ("row", "paired", "frozen_core", "load_data", "check", "verdict")}
    ly.install()
    assert a3.wait_for_memory is ly.wait_for_memory
    assert a3.score is ly.score
    assert {k: getattr(a3, k) for k in other} == other


def test_memory_rule_polls_and_yields(patched, monkeypatch):
    mx = 100 * GIB
    seq = iter([(70 * GIB, 30 * GIB, mx), (70 * GIB, 30 * GIB, mx), (50 * GIB, 30 * GIB, mx)])
    monkeypatch.setattr(a3, "cgroup", lambda: next(seq))
    ly.wait_for_memory(24.0, "fit_full")          # 70 + 24 >= 80 twice, then 50 + 24 < 80
    assert patched["sleep"] == [120, 120]
    assert patched["yield"] == 3                  # before every check
    assert patched["orig_wait"] == [(24.0, "fit_full")]
    assert [y["point"] for y in ly.YIELDS] == ["before fit_full"]
    assert a3.METRICS["memory.before.fit_full.launcher_waited_s"]["value"] == 240


def test_memory_rule_passes_at_once(patched, monkeypatch):
    monkeypatch.setattr(a3, "cgroup", lambda: (10 * GIB, 5 * GIB, 100 * GIB))
    ly.wait_for_memory(24.0, "fit_5M")
    assert patched["sleep"] == []
    assert patched["orig_wait"] == [(24.0, "fit_5M")]
    assert "memory.before.fit_5M.launcher_waited_s" not in a3.METRICS


def test_score_passes_through(patched):
    out = ly.score("core", "ev", "lnp", "ks", feats="feats")
    assert out == {"core": "core", "ev": "ev", "lnp": "lnp", "ks": "ks", "feats": "feats"}
    assert patched["yield"] == 1 and patched["orig_score"] == 1
    out = ly.score("c2", "e2", "l2", "k2")
    assert out["feats"] is None and patched["yield"] == 2
    assert [y["point"] for y in ly.YIELDS] == ["validation scoring"]


def test_no_lock_means_no_yield(monkeypatch):
    monkeypatch.delenv("TMLM_GPU_LOCK_MODE", raising=False)
    monkeypatch.setattr(ly, "YIELDS", [])
    assert ly.give_gpu_if_requested("anywhere") is False
    assert ly.YIELDS == []
