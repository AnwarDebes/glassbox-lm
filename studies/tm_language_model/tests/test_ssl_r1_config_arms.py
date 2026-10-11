"""Tests of experiments/ssl_r1_config_arms.py (the exploratory configuration arms of docs/13 amendment A8 item 2:
A3s8, A3s32, A3pos on A3's data, twin A3T of the control study, the A8 item 6 diagnostics). CPU engine and oracle
only, tiny corpora; the one CUDA test skips without a device and is run under scripts/gpu_run.sh shared.

(a) registration happens only through register() and never at import (the R1 arm list stays intact);
(b) the overrides reach the EngineConfig and the engines' parameter structs, and nothing else changes; the data equal A3's;
(c) budget 3 caps included literals at 3 (and the cap binds); p_neg 0 gives no negated literal (A3 does insert them);
(d) A3T is the twin of every arm: identical config for the s arms, and for A3pos an identical trajectory (state
    hash for state hash, oracle and CPU engine) although budget and neg_budget differ;
(e) the item 6 diagnostics against brute force; the train stage end to end; the report comparisons; the config."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _q in (ROOT / "experiments", ROOT / "src"):
    if str(_q) not in sys.path:
        sys.path.insert(0, str(_q))

import ssl_r1_config_arms as CA  # noqa: E402
import ssl_r1_controls as SC  # noqa: E402
import ssl_r1_pretext as R  # noqa: E402
from tmlm.data.synthetic import toy_corpus  # noqa: E402
from tmlm.engine.checkpoint import load_checkpoint  # noqa: E402
from tmlm.engine.config import INCLUDE_THRESHOLD, LIT_EMPTY, derive  # noqa: E402
from tmlm.engine.cpu.cpu_engine import CpuEngine  # noqa: E402
from tmlm.engine.oracle.oracle import OracleEngine  # noqa: E402
from tmlm.engine.rng import inv_q32  # noqa: E402
from tmlm.ssl import pretext as P  # noqa: E402

CJ = SC.load_config(CA.DEFAULT_CONFIG)
TINY = dict(vocab=40, n_clauses=64, batch=16, sweep_period=8, n_bag_k=2, dead_sweeps=2)
R1_ARMS = ["A0", "A1", "A2", "A3", "A4", "A5", "A6"]
EXPECTED = {"A3s8": {"s": 8.0}, "A3s32": {"s": 32.0}, "A3pos": {"budget": 3, "p_neg": 0.0, "neg_budget": 0}}


@pytest.fixture
def arms(monkeypatch):
    """The arms registered into COPIES of the two registries (restored after the test)."""
    monkeypatch.setattr(P, "ARMS", dict(P.ARMS))
    monkeypatch.setattr(SC, "CONTROL_ARMS", dict(SC.CONTROL_ARMS))
    return CA.register(CJ)


def _params(**over) -> dict:
    p = copy.deepcopy(CJ["params"])
    p["young_ladder"] = {**p["young_ladder"], "abs_batches": [2, 3], "percent_of_epoch1": [25, 50]}
    p["young_fraction"] = 0.3
    p["diagnostics"] = {**p["diagnostics"], "class_sum_rows": 40, "fired_chunk_rows": 7, "class_sum_bin_width": 8}
    p.update(over)
    return p


def _train(arm: str, corpus, tmp: Path, epochs: int = 4, p: dict | None = None, seed: int = 1):
    """SC.train_ladder (the control study's loop, CPU engine) plus the metrics a run would hold."""
    p = p or _params()
    base, twin = SC.CONTROL_ARMS[arm]
    ad = P.arm_data(base, corpus, vocab=40)
    cfg = SC.arm_config(arm, seed, p, **TINY)
    rec = R.Rec(smoke=True)
    tmp.mkdir(parents=True, exist_ok=True)
    h = SC.train_ladder(ad, cfg, epochs, tmp, "cpu", rec, p, twin, yield_gpu=False)
    rec.m("control.arm", arm)
    rec.m("control.seed", seed)
    (tmp / "metrics.json").write_text(json.dumps(rec.metrics, default=str))
    return ad, cfg, rec, h


def _states(run: Path) -> dict[str, dict]:
    return {n: load_checkpoint(run / f"checkpoint_{n}.npz")[1] for n in SC.checkpoint_names(run)}


def _run_epochs(cfg, ad, epochs: int, kind: str) -> tuple[list[str], dict]:
    """Per-epoch state hashes and summed counters, one engine per epoch (F2 example sets), CPU engine or oracle."""
    eng, hashes, tot = None, [], {}
    for e in range(epochs):
        st = None if eng is None else eng.state()
        eng = CpuEngine(cfg, corpus=ad.corpus(e)) if kind == "cpu" else OracleEngine(cfg, corpus=ad.corpus(e))
        if st is not None:
            eng.load_state(st)
        for b in range(eng.n_batches()):
            for k, v in eng.train_batch(e, b).items():
                tot[k] = tot.get(k, 0) + int(v)
        hashes.append(eng.state_hash())
    return hashes, tot


# ------------------------------------------------------------------ (a) registration
def test_import_registers_nothing():
    assert sorted(P.ARMS) == R1_ARMS
    assert set(SC.CONTROL_ARMS) == {"A3", "A0", "A3T", "A0T"}


def test_register_adds_the_arms_idempotently_and_refuses_conflicts(arms, monkeypatch):
    assert set(arms) == set(EXPECTED)
    for name, arm in arms.items():
        assert P.ARMS[name] is arm and SC.CONTROL_ARMS[name] == (name, False)
        a3 = P.ARMS["A3"]
        assert (arm.mode, arm.f1, arm.f2, arm.random_kind_map) == (a3.mode, a3.f1, a3.f2, a3.random_kind_map)
        assert arm.to_dict()["engine_overrides"] == EXPECTED[name]
    assert CA.register(CJ) == arms                       # idempotent
    bad = copy.deepcopy(CJ)
    bad["params"]["config_arms"]["A3s8"]["engine_overrides"]["s"] = 9
    with pytest.raises(SystemExit):
        CA.register(bad)                                  # a different arm under a registered name
    bad2 = copy.deepcopy(CJ)
    bad2["params"]["config_arms"] = {"A3sT": {"data_arm": "A3", "engine_overrides": {"s": 8}, "description": "x"}}
    with pytest.raises(SystemExit):
        CA.arm_entries(bad2)                              # a name ending in T means a twin in the control study
    bad3 = copy.deepcopy(CJ)
    bad3["params"]["config_arms"]["A3pos"]["engine_overrides"]["budget"] = 3.5
    with pytest.raises(SystemExit):
        CA.arm_entries(bad3)                              # integer field


# ------------------------------------------------------------------ (b) overrides reach the engine config; data equal A3
def test_overrides_reach_the_engine_config_and_nothing_else_changes(arms):
    p = CJ["params"]
    c = toy_corpus(vocab=40, n_docs=2, doc_len=40, seed=3)
    for name, ov in EXPECTED.items():
        for seed in (100, 101):
            cfg, ref = SC.arm_config(name, seed, p), P.engine_config("A3", seed)
            diff = {k: v for k, v in cfg.to_dict().items() if ref.to_dict()[k] != v}
            assert diff == ov
            assert CA.override_diff(name, seed, p) == {k: [v, ref.to_dict()[k]] for k, v in ov.items()}
        tiny = SC.arm_config(name, 1, p, **TINY)
        d = derive(tiny)
        cpu = CpuEngine(tiny, corpus=c).params
        orc = OracleEngine(tiny, corpus=c)
        if "s" in ov:
            s = int(ov["s"])
            assert d.inv_s_q32 == inv_q32(ov["s"]) == (1 << 32) // s
            assert (cpu.s_num, cpu.s_den, cpu.inv_s_q32) == (s, 1, (1 << 32) // s)
            assert (orc.s_num, orc.s_den) == (s, 1)
            assert (cpu.BUDGET, cpu.neg_budget) == (12, 2)
        else:
            assert (cpu.BUDGET, cpu.neg_budget, cpu.p_neg_q32) == (3, 0, 0)
            assert d.p_neg_q32 == 0 and orc.cfg.budget == 3 and orc.cfg.neg_budget == 0
            assert (cpu.s_num, cpu.s_den) == (128, 1) and cpu.p_ins_q32 == derive(SC.arm_config("A3", 1, p, **TINY)).p_ins_q32


def test_config_arm_data_equal_A3(arms, monkeypatch):
    monkeypatch.setattr(P, "F2_M", 5)
    c = toy_corpus(vocab=40, n_docs=4, doc_len=80, seed=5)
    ref = P.arm_data("A3", c, vocab=40)
    assert not np.array_equal(ref.targets(0), ref.targets(1))   # F2 redraws per epoch: the comparison is not vacuous
    for name in EXPECTED:
        ad = P.arm_data(name, c, vocab=40)
        assert ad.targets_vary_by_epoch and ad.info == ref.info
        assert np.array_equal(ad.kind_map, ref.kind_map) and np.array_equal(ad.sym_freq, ref.sym_freq)
        for e in range(4):
            assert np.array_equal(ad.targets(e), ref.targets(e))
        rec = R.Rec(smoke=True)
        assert CA.data_check(name, c, _params(vocab=40), rec) and rec.failed == []


# ------------------------------------------------------------------ (c) budget 3 and p_neg 0
def test_budget_3_caps_included_literals_at_3_and_the_cap_binds(arms, tmp_path, monkeypatch):
    monkeypatch.setattr(P, "F2_M", 5)
    c = toy_corpus(vocab=40, n_docs=6, doc_len=120, seed=5)
    _, cfg, rec, h = _train("A3pos", c, tmp_path / "pos")
    assert cfg.budget == 3 and rec.failed == []
    blocked = at_budget = 0
    for name, st in _states(tmp_path / "pos").items():
        caps = CA.literal_caps(st, cfg)
        assert caps["max_included"] <= 3 and CA.caps_ok(caps), (name, caps)
        alive = st["alive"].astype(bool)
        lid, ls = st["lit_id"][alive], st["lit_state"][alive]
        n_inc = ((lid != LIT_EMPTY) & (ls >= INCLUDE_THRESHOLD)).sum(axis=1)
        assert int(n_inc.max()) <= 3
        at_budget += int((n_inc == 3).sum())
        # a literal waiting at 127 in a clause at the budget: the 127 -> 128 crossing was suppressed
        blocked += int(((n_inc >= 3)[:, None] & (lid != LIT_EMPTY) & (ls == INCLUDE_THRESHOLD - 1)).any(axis=1).sum())
    assert at_budget > 0 and blocked > 0
    # the same data under A3 (budget 12) grows clauses beyond 3 literals: the cap is what stops them
    _, cfg3, _, _ = _train("A3", c, tmp_path / "a3")
    assert max(CA.literal_caps(st, cfg3)["max_included"] for st in _states(tmp_path / "a3").values()) > 3
    # the oracle agrees with the CPU engine bit for bit under the A3pos config and keeps its invariants
    ad = P.arm_data("A3pos", c, vocab=40)
    hc, _ = _run_epochs(cfg, ad, 2, "cpu")
    eng = None
    for e in range(2):
        st = None if eng is None else eng.state()
        eng = OracleEngine(cfg, corpus=ad.corpus(e))
        if st is not None:
            eng.load_state(st)
        for b in range(eng.n_batches()):
            eng.train_batch(e, b)
        assert eng.state_hash() == hc[e]
    assert eng.check_invariants() == []


def test_p_neg_0_gives_no_negated_literal(arms, tmp_path, monkeypatch):
    monkeypatch.setattr(P, "F2_M", 5)
    c = toy_corpus(vocab=40, n_docs=6, doc_len=120, seed=5)
    _, cfg, _, h = _train("A3pos", c, tmp_path / "pos")
    assert cfg.p_neg == 0.0 and h["totals"].get("n_neg_inserted", 0) == 0
    for name, st in _states(tmp_path / "pos").items():
        tracked = (st["lit_id"] != LIT_EMPTY) & st["alive"].astype(bool)[:, None]
        assert not np.any(tracked & ((st["lit_id"] & 1) == 1)), name
    _, cfg3, _, h3 = _train("A3", c, tmp_path / "a3")
    assert cfg3.p_neg > 0 and h3["totals"]["n_neg_inserted"] > 0
    assert any(CA.literal_caps(st, cfg3)["tracked_negated"] > 0 for st in _states(tmp_path / "a3").values())


def test_s_overrides_change_the_trajectory(arms, monkeypatch):
    monkeypatch.setattr(P, "F2_M", 5)
    c = toy_corpus(vocab=40, n_docs=4, doc_len=80, seed=7)
    ad = P.arm_data("A3", c, vocab=40)
    p = _params()
    ref, _ = _run_epochs(SC.arm_config("A3", 1, p, **TINY), ad, 2, "cpu")
    for name in EXPECTED:
        got, _ = _run_epochs(SC.arm_config(name, 1, p, **TINY), P.arm_data(name, c, vocab=40), 2, "cpu")
        assert got[-1] != ref[-1], name


# ------------------------------------------------------------------ (d) A3T is the twin of every arm
def test_twin_relation_of_every_arm(arms):
    p = CJ["params"]
    a3t = SC.arm_config("A3T", 100, p)
    for name in ("A3s8", "A3s32"):
        rel = CA.twin_relation(name, 100, p)
        assert rel["exact"] and rel["equivalent"] and rel["literal_thresholds_zero"]
        assert CA.twin_config_of(name, 100, p) == a3t
    rel = CA.twin_relation("A3pos", 100, p)
    assert not rel["exact"] and rel["equivalent"]
    assert rel["differing_fields"] == {"budget": [3, 12], "neg_budget": [0, 2]}
    t = SC.twin_thresholds(CA.twin_config_of("A3pos", 100, p))
    assert (t["p_ins_q32"], t["p_neg_q32"], t["inv_s_q32"], t["s_num"], t["s_den"]) == (0, 0, 0, 1 << 62, 1)


def test_twin_relation_refuses_fields_a_twin_does_read(arms, monkeypatch):
    p = copy.deepcopy(CJ["params"])
    p["config_arms"] = {**p["config_arms"], "A3b2": {"data_arm": "A3", "engine_overrides": {"budget": 2}, "description": "x"},
                        "A3kc": {"data_arm": "A3", "engine_overrides": {"k_cover": 4}, "description": "x"}}
    cj = {**CJ, "params": p}
    CA.register(cj)
    assert not CA.twin_relation("A3b2", 100, p)["equivalent"]   # a budget below the 3 seed literals
    assert not CA.twin_relation("A3kc", 100, p)["equivalent"]   # seeding differs: not A3T's trajectory


@pytest.mark.parametrize("kind", ["cpu", "oracle"])
def test_A3pos_twin_has_the_trajectory_of_A3T(arms, monkeypatch, kind):
    monkeypatch.setattr(P, "F2_M", 5)
    c = toy_corpus(vocab=40, n_docs=6, doc_len=120, seed=11)
    ad = P.arm_data("A3", c, vocab=40)
    p = _params()
    tw = CA.twin_config_of("A3pos", 1, p, **TINY)
    a3t = SC.arm_config("A3T", 1, p, **TINY)
    assert (tw.budget, tw.neg_budget, a3t.budget, a3t.neg_budget) == (3, 0, 12, 2)
    epochs = 3
    h_tw, t_tw = _run_epochs(tw, ad, epochs, kind)
    h_t, t_t = _run_epochs(a3t, ad, epochs, kind)
    assert h_tw == h_t and t_tw == t_t
    assert t_t["n_seeded"] > 0 and t_t["n_recycled"] > 0 and t_t.get("n_inserted", 0) == 0
    # with literal learning on, budget 3 does change the trajectory (the equality above is not vacuous)
    h_pos, _ = _run_epochs(SC.arm_config("A3pos", 1, p, **TINY), ad, epochs, kind)
    h_a3, _ = _run_epochs(SC.arm_config("A3", 1, p, **TINY), ad, epochs, kind)
    assert h_pos[-1] != h_a3[-1]


# ------------------------------------------------------------------ (e) diagnostics
def test_e_bound_and_owned_per_class():
    assert CA.e_bound(64) == 174 and math.floor(174 / math.e) >= 64 > math.floor(173 / math.e)
    assert CA.e_bound(1) == 3
    st = {"alive": np.array([1, 1, 1, 0], np.uint8),
          "vote_cls": np.array([[0, 2, 0xFFFF], [2, 1, 0xFFFF], [2, 0xFFFF, 0xFFFF], [0, 1, 2]], np.uint16),
          "vote_w": np.array([[3, -1, 0], [5, 2, 0], [1, 0, 0], [9, 9, 9]], np.int16)}
    share = np.array([0.5, 0.25, 0.25, 0.0])
    o = CA.owned_per_class(st, 4, 1, share)
    # class 0: clause 0; class 1: clause 1; class 2: clauses 1 and 2 (clause 0 votes -1, clause 3 is dead); class 3: none
    assert o["owned_total"] == 4 and o["max"] == 2
    assert o["frac_classes_zero"] == 0.25 and o["frac_classes_ge_T"] == 0.75 and o["frac_classes_ge_2T"] == 0.25
    assert o["frac_classes_ge_eT"] == 0.0 and o["bound_eT"] == 3
    assert o["hist_log2"][:3] == [1, 2, 1]
    assert o["target_share_ge_T"] == 1.0 and o["target_share_ge_2T"] == 0.25 and o["target_share_zero"] == 0.0


def test_include_crossings_on_a_hand_built_pair():
    L = 6
    E = LIT_EMPTY

    def view(alive, birth, ids, states):
        return {"alive": np.array(alive, bool), "birth": np.array(birth, np.uint32),
                "lit_id": np.array(ids, np.uint32), "lit_state": np.array(states, np.uint8)}
    # clause 0 (same birth): 10 excluded -> included, 12 included -> excluded, 14 included -> freed,
    #   16 excluded -> gone, 18 new included, 20 new excluded; clause 1: other birth (recycled), ignored;
    #   clause 2: dead now, ignored
    prev = view([1, 1, 1], [5, 5, 5],
                [[10, 12, 14, 16, E, E], [10, 12, E, E, E, E], [10, E, E, E, E, E]],
                [[127, 130, 128, 60, 0, 0], [127, 130, 0, 0, 0, 0], [200, 0, 0, 0, 0, 0]])
    cur = view([1, 1, 0], [5, 9, 0],
               [[10, 12, 18, 20, E, E], [10, 14, E, E, E, E], [E] * L],
               [[128, 127, 140, 123, 0, 0], [128, 128, 0, 0, 0, 0], [0] * L])
    cr = CA.include_crossings(prev, cur)
    assert cr["n_common"] == 1 and cr["tracked_prev"] == 4 and cr["included_prev"] == 2
    assert (cr["same_literal_up"], cr["same_literal_down"], cr["included_then_freed"], cr["inserted_then_included"]) == (1, 1, 1, 1)
    assert cr["excluded_then_gone"] == 1 and cr["inserted_still_excluded"] == 1
    assert cr["net_crossings"] == 4 and cr["net_crossings_per_tracked_prev"] == 1.0
    assert CA.include_crossings(None, cur)["n_common"] == 0


def _tiny_val(seed: int = 21):
    val = toy_corpus(vocab=40, n_docs=3, doc_len=70, seed=seed)
    targets = np.arange(1, val.tok.shape[0], 3, dtype=np.int64)
    hidden = np.zeros(val.tok.shape[0], np.uint8)
    hidden[targets] = 1
    return val, targets, hidden


def test_class_sum_stats_equal_brute_force(arms, tmp_path, monkeypatch):
    monkeypatch.setattr(P, "F2_M", 5)
    c = toy_corpus(vocab=40, n_docs=6, doc_len=120, seed=5)
    _, cfg, _, _ = _train("A3s8", c, tmp_path / "run")
    _, st, _ = load_checkpoint(tmp_path / "run" / "checkpoint_epoch4.npz")
    val, targets, hidden = _tiny_val()
    eng = CpuEngine(cfg, corpus=val)
    eng.load_state(st)
    rows = targets[::2]
    gold = np.asarray(val.tok)[rows].astype(np.int64)
    cs, same = CA.class_sum_stats(eng, cfg, st, rows, gold, hidden, 8, 5)
    assert same
    T = cfg.T
    sy, top, cover, fired = [], [], [], []
    for r, g in zip(rows.tolist(), gold.tolist()):
        f = eng.fired_clauses(4, r, hidden).tolist()
        s = np.zeros(40, np.int64)
        cv = 0
        for j in f:
            for k in range(cfg.v_slots):
                if st["vote_cls"][j, k] != 0xFFFF:
                    s[st["vote_cls"][j, k]] += int(st["vote_w"][j, k])
                    cv += int(st["vote_cls"][j, k] == g and st["vote_w"][j, k] > 0)
        sy.append(s[g])
        top.append(max(s[cc] for cc in range(40) if cc != g))
        cover.append(cv)
        fired.append(len(f))
    sy, top = np.array(sy), np.array(top)
    assert np.any(sy != 0)                                           # the bank does vote on these rows
    assert sum(cs["target_sum_hist"]) == rows.shape[0] == cs["rows"] and sum(cs["all_class_sums_hist"]) == rows.shape[0] * 40
    assert cs["bin_edges"][0] == -T and cs["bin_edges"][-1] == T + 1
    assert cs["target_sum_hist"] == np.histogram(np.clip(sy, -T, T), bins=cs["bin_edges"])[0].tolist()
    assert cs["frac_target_sum_ge_T"] == float((sy >= T).mean()) and cs["frac_target_sum_le_0"] == float((sy <= 0).mean())
    assert cs["frac_target_above_every_other"] == float((sy > top).mean())
    assert cs["cover_mean"] == float(np.mean(cover)) and cs["fired_per_row_mean"] == float(np.mean(fired))


def test_diag_checkpoints_over_a_train_run(arms, tmp_path, monkeypatch):
    monkeypatch.setattr(P, "F2_M", 5)
    c = toy_corpus(vocab=40, n_docs=6, doc_len=120, seed=5)
    p = _params()
    ad, cfg, trec, h = _train("A3pos", c, tmp_path / "run", p=p)
    val, targets, hidden = _tiny_val()
    val = P.with_tables(val, kind_map=ad.kind_map, sym_freq=ad.sym_freq)
    sample = np.arange(0, targets.shape[0], 2)
    rec = R.Rec(smoke=True)
    out = CA.diag_checkpoints(tmp_path / "run", "A3pos", 1, p, val, targets, hidden, sample, rec, "cpu", **TINY)
    assert rec.failed == [], rec.failed
    names = SC.checkpoint_names(tmp_path / "run")
    assert list(out) == names and names[-1] == "epoch4"
    states = _states(tmp_path / "run")
    for i, name in enumerate(names):
        d = out[name]
        assert d["literals"]["included_per_clause_max"] <= 3 and d["literals"]["frac_literals_negated"] in (0.0, None)
        assert d["owned_per_class"] == CA.owned_per_class(states[name], 40, cfg.T, np.bincount(np.asarray(val.tok)[targets], minlength=40) / targets.shape[0])
        if i:
            assert d["include_crossings"] == {**CA.include_crossings(SC.literal_view(states[names[i - 1]]), SC.literal_view(states[name])),
                                              "previous": names[i - 1]}
    per = rec.metrics["diag.include_crossings_per_epoch"]["value"]
    assert set(per) == {"1", "2", "3", "4"}
    ev = trec.metrics["train.epoch3.n_type1"]["value"] + trec.metrics["train.epoch3.n_type2"]["value"]
    assert per["3"]["feedback_events"] == ev and per["3"]["intervals"] == ["epoch2->epoch3"]
    assert per["3"]["net_crossings"] == out["epoch3"]["include_crossings"]["net_crossings"]
    assert per["1"]["intervals"][0] == f"{names[0]}->{names[1]}" and "feedback_events" not in per["1"]
    assert per["1"]["intervals"][-1].endswith("->epoch1")


# ------------------------------------------------------------------ (e) stages
def test_train_stage_end_to_end(arms, tmp_path, monkeypatch):
    """stage_train on a toy prefix: the checks it adds pass and ssl_r1_controls.stage_train runs unchanged."""
    monkeypatch.setattr(P, "F2_M", 5)
    c = toy_corpus(vocab=40, n_docs=4, doc_len=80, seed=5)
    monkeypatch.setattr(R, "load_prefix", lambda p, rec: c)
    run = tmp_path / "train"
    run.mkdir()
    monkeypatch.setattr(SC, "out_dir", lambda stage: run)
    cj = copy.deepcopy(CJ)
    cj["params"].update(vocab=40, n_clauses=64, epochs=2, young_fraction=0.5)
    cj["params"]["young_ladder"] = {**cj["params"]["young_ladder"], "abs_batches": [1], "percent_of_epoch1": [50]}
    for arm in ("A3pos", "A3s8"):
        rec = R.Rec(smoke=True)
        CA.stage_train(argparse.Namespace(arm=arm, seed=100, engine="cpu"), cj, rec)
        assert rec.failed == [], rec.failed
        m = {k: v["value"] for k, v in rec.metrics.items()}
        assert m["check.config_arm.overrides_reach_engine_config"] and m["check.twin.A3T_serves_as_twin"]
        assert m["check.config_arm.data_equal_A3"] and m["check.config_arm.literal_caps.epoch2"]
        names = SC.checkpoint_names(run)
        assert "young_b1" in names and "c0y" in names
        for name in names:                                  # every checkpoint: config checked, then its caps
            assert m[f"check.config_arm.checkpoint_config_equals_arm.{name}"] and m[f"check.config_arm.literal_caps.{name}"]
            assert rec.metrics[f"check.config_arm.literal_caps.{name}"]["budget"] == SC.arm_config(arm, 100, cj["params"]).budget
        assert m["control.arm"] == arm and m["control.twin"] is False and m["replay.reference"] is None
        for k, v in EXPECTED[arm].items():
            assert m["engine_config"][k] == v
        assert m["arm"]["engine_overrides"] == EXPECTED[arm] and m["arm"]["f2"] is True
        assert m["twin.relation"]["exact"] == (arm != "A3pos")


def test_train_stage_checks_each_checkpoint_config_against_the_arm(arms, tmp_path, monkeypatch):
    """A checkpoint whose config is not the arm's intended EngineConfig fails its config check and its caps check
    (the caps are those of the intended config, so a checkpoint cannot pass by carrying a looser budget)."""
    monkeypatch.setattr(P, "F2_M", 5)
    c = toy_corpus(vocab=40, n_docs=4, doc_len=80, seed=5)
    monkeypatch.setattr(R, "load_prefix", lambda p, rec: c)
    run = tmp_path / "train"
    run.mkdir()
    monkeypatch.setattr(SC, "out_dir", lambda stage: run)
    cj = copy.deepcopy(CJ)
    cj["params"].update(vocab=40, n_clauses=64, epochs=2, young_fraction=0.5)
    cj["params"]["young_ladder"] = {**cj["params"]["young_ladder"], "abs_batches": [1], "percent_of_epoch1": [50]}
    real = SC.stage_train

    def train_then_move_the_budget(args, cj_, rec):
        real(args, cj_, rec)
        P.ARMS["A3pos"] = P.Arm("A3pos", P.ARMS["A3pos"].mode, f1=P.ARMS["A3pos"].f1, f2=P.ARMS["A3pos"].f2,
                                random_kind_map=P.ARMS["A3pos"].random_kind_map,
                                engine=tuple(P.ARMS["A3"].engine) + (("budget", 2), ("neg_budget", 0), ("p_neg", 0.0)))

    monkeypatch.setattr(SC, "stage_train", train_then_move_the_budget)
    rec = R.Rec(smoke=True)
    CA.stage_train(argparse.Namespace(arm="A3pos", seed=100, engine="cpu"), cj, rec)
    names = SC.checkpoint_names(run)
    for name in names:
        assert f"config_arm.checkpoint_config_equals_arm.{name}" in rec.failed
        assert f"config_arm.literal_caps.{name}" in rec.failed
        assert rec.metrics[f"check.config_arm.checkpoint_config_equals_arm.{name}"]["differing"] == {"budget": [3, 2]}
        assert rec.metrics[f"check.config_arm.literal_caps.{name}"]["budget"] == 2


def test_census_stage_end_to_end(arms, tmp_path, monkeypatch):
    """CA.stage_census on a tiny train run of stage_train, with a toy validation split, a toy frozen eval set and a
    counted test-access log swapped in: the control study's census and the item 6 diagnostics of every checkpoint,
    the dataset hashes, no test access."""
    import tmlm.data.corpus_store as CS
    import tmlm.data.evalsets as EV

    monkeypatch.setattr(P, "F2_M", 100)  # several batches of 1,024 in epoch 1: the young rungs and c0y precede epoch1
    c = toy_corpus(vocab=40, n_docs=40, doc_len=200, seed=5)
    monkeypatch.setattr(R, "load_prefix", lambda p, rec: c)
    run = tmp_path / "train"
    run.mkdir()
    monkeypatch.setattr(SC, "out_dir", lambda stage: run)
    cj = copy.deepcopy(CJ)
    cj["params"].update(vocab=40, n_clauses=64, epochs=2, young_fraction=0.5)
    cj["params"]["young_ladder"] = {**cj["params"]["young_ladder"], "abs_batches": [1], "percent_of_epoch1": [50]}
    cj["params"]["diagnostics"] = {**cj["params"]["diagnostics"], "class_sum_rows": 40, "fired_chunk_rows": 7,
                                   "class_sum_bin_width": 8, "engine": "cpu"}
    trec = R.Rec(smoke=True)
    CA.stage_train(argparse.Namespace(arm="A3pos", seed=100, engine="cpu"), cj, trec)
    assert trec.failed == [], trec.failed
    (run / "metrics.json").write_text(json.dumps(trec.metrics, default=str))   # what run_logged would hold

    val, targets, hidden = _tiny_val()
    vdir = tmp_path / "validation"
    vdir.mkdir()
    (vdir / "meta.json").write_text(json.dumps({"sha256": {"tok.npy": "tok-sha"}}))
    accesses: list[str] = []          # the counted test-access log: one entry per test-split load
    calls: list[str] = []

    def fake_load_split(corpus, vocab, split, **kw):
        assert (corpus, vocab) == (R.CORPUS, 40)
        if split == "test":
            accesses.append("load_split")
        return val

    def fake_load_evalset(corpus, vocab, split, variant, **kw):
        assert (corpus, vocab, variant) == (R.CORPUS, 40, "subword")
        if split == "test":
            accesses.append("load_evalset")
        return {"targets": targets, "hidden": hidden, "meta": {"sha256_combined": "evalset-sha"}}

    def fake_count(root=None):
        calls.append("count")
        return 7 + len(accesses)

    hashes: dict = {}
    monkeypatch.setattr(CS, "load_split", fake_load_split)
    monkeypatch.setattr(CS, "split_dir", lambda corpus, vocab, split, root=None: vdir if split == "validation" else tmp_path / split)
    monkeypatch.setattr(EV, "load_evalset", fake_load_evalset)
    monkeypatch.setattr(EV, "test_access_count", fake_count)
    monkeypatch.setattr(R, "dataset_hashes", lambda d: hashes.update(d))

    rec = R.Rec(smoke=True)
    CA.stage_census(argparse.Namespace(), cj, rec, run)
    assert rec.failed == [], rec.failed
    m = {k: v["value"] for k, v in rec.metrics.items()}
    names = SC.checkpoint_names(run)
    assert names[-1] == "epoch2" and names[0] == "young_b1" and names.index("c0y") < names.index("epoch1")
    assert m["control.arm"] == "A3pos" and m["control.seed"] == 100 and m["census.train_run"] == str(run)
    assert hashes == {f"{R.CORPUS}_bpe40_validation_tok": "tok-sha", "evalset_validation_subword_V40": "evalset-sha"}
    assert calls == ["count", "count"] and accesses == []
    assert m["check.diag.no_test_access"] and m["diag.test_accesses_before"] == m["diag.test_accesses_after"] == 7
    assert m["diag.validation_targets"] == targets.shape[0] and m["diag.class_sum_sample_rows"] == 40
    assert m["diag.evalset_sha256_combined"] == "evalset-sha"
    states = _states(run)
    share = np.bincount(np.asarray(val.tok)[targets], minlength=40)[:40] / targets.shape[0]
    cfg = SC.arm_config("A3pos", 100, cj["params"])
    for name in names:
        assert m[f"check.{name}.state_hash_equals_train_run"]
        assert m[f"census.{name}"]["alive"] == int(states[name]["alive"].sum())
        assert m[f"check.diag.{name}.target_sum_rebuilt_equals_predict_scores"]
        assert m[f"diag.{name}.literals"]["included_per_clause_max"] <= 3
        assert m[f"diag.{name}.owned_per_class"] == CA.owned_per_class(states[name], 40, cfg.T, share)
        assert m[f"diag.{name}.class_sums"]["rows"] == 40
        assert sum(m[f"diag.{name}.class_sums"]["target_sum_hist"]) == 40
    per = m["diag.include_crossings_per_epoch"]
    assert set(per) == {"1", "2"} and per["2"]["intervals"] == ["epoch1->epoch2"]
    assert per["2"]["feedback_events"] == trec.metrics["train.epoch2.n_type1"]["value"] + trec.metrics["train.epoch2.n_type2"]["value"]


def test_main_refuses_arms_and_seeds_outside_the_study(arms):
    for argv in (["--stage", "train", "--arm", "A3", "--seed", "100"], ["--stage", "train", "--arm", "A3s8", "--seed", "7"],
                 ["--stage", "train", "--arm", "A3T", "--seed", "100"]):
        with pytest.raises(SystemExit):
            CA.main(argv)


def _cells(fh, arm: str, seed: int, name: str, q0: float, codes=("all", "dim")) -> None:
    for code in codes:
        for d in range(5):
            for task in ("agnews", "sst2", "trec"):
                fh.write(json.dumps({"arm": arm, "seed": seed, "set": f"{name}.{code}", "clf": "lr", "task": task,
                                     "budget": 32, "draw": d, "acc": q0 + 0.001 * d}) + "\n")


def test_report_comparisons_through_main(arms, tmp_path, monkeypatch):
    monkeypatch.setenv("SSL_SCRATCH", str(tmp_path / "scratch"))
    acc = {("A3s8", "epoch4"): 0.63, ("A3s8", "epoch1"): 0.62, ("A3s8", "c0y"): 0.60, ("A3s8", "young_b382"): 0.61,
           ("A3T", "epoch4"): 0.615, ("A3", "epoch4"): 0.625, ("A3", "c0y"): 0.60, ("A3", "young_b196"): 0.628}
    d = tmp_path / "probe"
    d.mkdir()
    with open(d / "cells.jsonl", "w") as fh:
        for (a, name), q0 in acc.items():
            _cells(fh, a, 100, name, q0)
    qd = tmp_path / "qpred"
    qd.mkdir()
    (qd / "metrics.json").write_text(json.dumps({"control.arm": {"value": "A3s8"}, "control.seed": {"value": 100},
                                                 "qpred.epoch4.loglin_top1": {"value": 0.31}, "qpred.c0y.loglin_top1": {"value": 0.2}}))
    rc = CA.main(["--stage", "report", "--probe-runs", str(d), "--qpred-runs", str(qd)])
    assert rc == 0
    out = next((tmp_path / "scratch").glob("ssl_r1c_report_*"))
    s = json.loads((out / "summary.json").read_text())
    x = s["A3s8.seed100.all.budget32"]
    assert x["max_rung"] == "young_b382.all" and x["ref_max_rung"] == "young_b196.all"
    for key, want in (("d_ladder", 2.0), ("d_twin", 1.5), ("d_vs_A3", 0.5), ("d_vs_A3_ladder", 0.2), ("d_epoch4_vs_epoch1", 1.0)):
        assert abs(x[key] - want) < 1e-9, key
        assert x[key + "_bootstrap"]["n"] == 5
    assert x["q_pred_bank"] == 0.31 and x["q_pred_c0y"] == 0.2 and x["q_pred_twin"] is None
    assert s["A3pos.seed100.all.budget32"]["q_bank"] is None                         # no rows: reported as missing
    assert set(s) == {f"{a}.seed{sd}.{c}.budget{b}" for a in EXPECTED for sd in (100, 101) for c in ("all", "dim") for b in (8, 32, 128)}


# ------------------------------------------------------------------ config
def test_config_is_exploratory_names_the_twin_and_shares_the_control_params():
    CA.check_params_equal_controls(CJ)
    SC.check_shared_params(CJ)
    assert "EXPLORATORY" in CJ["status"] and "DESIGN ONLY" in CJ["status"] and "A8 item 2" in CJ["status"]
    assert CJ["twin"]["arm"] == "A3T" and "whatever the arm's s is" in CJ["twin"]["why_valid"]
    assert {n: {k: v for k, v in a["engine_overrides"].items()} for n, a in CJ["params"]["config_arms"].items()} == EXPECTED
    assert all(a["data_arm"] == "A3" for a in CJ["params"]["config_arms"].values())
    p = CJ["params"]
    assert p["seeds"] == [100, 101] and p["probe_budgets"] == [8, 32, 128] and p["chain_checkpoints"]["features"] == "ladder,c0y,epoch1,epoch4"
    assert set(CJ["diagnostics_item6"]) == {"literals", "owned_per_class", "include_crossings", "class_sums"}
    assert "no verdict" in CJ["report"]["status"]
    for k in ("r1_config", "doc_set_reference", "params_shared_with_r1"):
        assert CJ[k] == SC.load_config(ROOT / CJ["controls_config"])[k]


def test_cuda_engine_receives_the_overrides_and_matches_the_cpu_engine(arms, monkeypatch):
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.Device(0).compute_capability
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"no CUDA device: {exc}")
    from tmlm.engine.cuda.cuda_engine import CudaEngine
    monkeypatch.setattr(P, "F2_M", 5)
    c = toy_corpus(vocab=40, n_docs=4, doc_len=80, seed=5)
    for name in ("A3pos", "A3s8"):
        cfg = SC.arm_config(name, 2, _params(), **TINY)
        ad = P.arm_data(name, c, vocab=40)
        hc, tc = _run_epochs(cfg, ad, 2, "cpu")
        eng = None
        tg: dict = {}
        for e in range(2):
            st = None if eng is None else eng.state()
            eng = CudaEngine(cfg, corpus=ad.corpus(e))
            if st is not None:
                eng.load_state(st)
            for b in range(eng.n_batches()):
                for k, v in eng.train_batch(e, b).items():
                    tg[k] = tg.get(k, 0) + int(v)
            assert eng.state_hash() == hc[e], (name, e)
        assert tg == tc
        prm = eng.params
        if name == "A3pos":
            assert (prm.BUDGET, prm.neg_budget, prm.p_neg_q32) == (3, 0, 0)
        else:
            assert (prm.s_num, prm.s_den) == (8, 1)
