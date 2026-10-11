"""experiments/ssl_r4_controls.py (docs/12 F29 items 2 and 3): the pure parts, on the CPU.

(a) the ladder rungs (C0y exactly as ssl_r4_images.py computes its young bank), (b) the replay comparison (a
perturbed hash is a failure), (c) the twin config and its derived thresholds, (d) the device-side slot transition
counts on hand-made snapshots, (e) the random bank drawn at given positions (identical to tmlm.ssl.confirm.random_bank
at every position) and its image codes against the pure-Python reference, (f) the per-bank diagnostics, (g) the
report statistics and the literal-learning label, end to end on synthetic runs, (h) the config files. The GPU stages
(replay, twin, randbank on cupy, diag) run in the smoke chain of the build task.
"""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _q in (ROOT / "src", ROOT / "experiments"):
    if str(_q) not in sys.path:
        sys.path.insert(0, str(_q))

import ssl_r1_controls as C1  # noqa: E402
import ssl_r4_controls as M  # noqa: E402
from tmlm.data.synthetic import make_kind_map  # noqa: E402
from tmlm.engine.config import INCLUDE_THRESHOLD, LIT_EMPTY, EngineConfig  # noqa: E402
from tmlm.ssl import confirm as CF  # noqa: E402
from tmlm.ssl import patch_tokens as PT  # noqa: E402

CONFIG = ROOT / "experiments" / "cvtm" / "configs" / "ssl_r4_controls.json"
SMOKE = ROOT / "experiments" / "cvtm" / "configs" / "ssl_r4_controls_smoke.json"


# ====================================================================== (a) ladder
def test_ladder_points_match_the_registered_young_bank_and_round_half_up():
    for nb in (1, 16, 99, 100, 490, 7639, 19531):
        pts = M.ladder_points(nb, 0.01, [5, 10])
        assert pts["c0y"] == max(1, int(math.ceil(0.01 * nb)))          # ssl_r4_images.stage_gpu young_b
        assert pts["pct5"] == max(1, int(math.floor(nb * 5 / 100 + 0.5)))
        assert pts["pct10"] == max(1, int(math.floor(nb * 10 / 100 + 0.5)))
        assert pts["epoch1"] == nb
    assert M.ladder_points(490, 0.01, [5, 10]) == {"c0y": 5, "pct5": 25, "pct10": 49, "epoch1": 490}
    assert M.unique_rungs(M.ladder_points(16, 0.01, [5, 10])) == {1: ["c0y", "pct5"], 2: ["pct10"], 16: ["epoch1"]}


# ====================================================================== (b) replay comparison
def _logged_and_ours():
    ep = {k: {"state_hash": f"h{k}", "alive": 10 * k, "dev_masked_top1": 0.5 + k / 100, "diag": {"n_examples": 7, "n_type1": k}}
          for k in (1, 2)}
    logged = {"engine_config": {"n_clauses": 8}, "targets.sha256": "t", "store_sha256": {"tok.npy": "x"},
              "main.young.state_hash": "hy", "main.young.dev_masked_top1": 0.3,
              "main.epoch1": {**ep[1], "wall_s": 1.0}, "main.epoch2": {**ep[2], "wall_s": 2.0},
              "main.final.codes": {"state_hash": "h2"}, "main.batches_per_epoch": 16}
    ours = {"engine_config": {"n_clauses": 8}, "targets_sha256": "t", "store_sha256": {"tok.npy": "x"},
            "c0y": {"batches": 1, "state_hash": "hy", "dev_masked_top1": 0.3}, "epochs": copy.deepcopy(ep), "final_state_hash": "h2"}
    return logged, ours


def test_replay_compare_passes_on_equal_values_and_fails_on_any_perturbation():
    logged, ours = _logged_and_ours()
    assert M.replay_failures(M.replay_compare(logged, ours, 2)) == []
    for path, val in ((("epochs", 2, "state_hash"), "hX"), (("c0y", "state_hash"), "hZ"), (("final_state_hash",), "h9"),
                      (("epochs", 1, "diag"), {"n_examples": 7, "n_type1": 2}), (("epochs", 1, "dev_masked_top1"), 0.1)):
        o = copy.deepcopy(ours)
        d = o
        for k in path[:-1]:
            d = d[k]
        d[path[-1]] = val
        bad = M.replay_failures(M.replay_compare(logged, o, 2))
        assert len(bad) == 1, (path, bad)
    lg = copy.deepcopy(logged)
    lg["main.epoch2"]["state_hash"] = "0" + lg["main.epoch2"]["state_hash"][1:]
    assert M.replay_failures(M.replay_compare(lg, ours, 2)) == ["epoch2.state_hash"]
    assert "epochs_logged" in M.replay_failures(M.replay_compare(logged, ours, 3))       # 3 epochs asked, 2 logged
    lg = copy.deepcopy(logged)
    del lg["main.young.state_hash"]
    assert "c0y.state_hash" in M.replay_failures(M.replay_compare(lg, ours, 2))           # missing is a failure


# ====================================================================== (c) twin
def test_twin_config_switches_off_every_literal_path_of_the_registered_engine():
    r4 = json.loads((ROOT / "experiments" / "cvtm" / "configs" / "ssl_r4_images.json").read_text())["params"]
    base = M.main_engine_config(r4, 2000, 1)
    base.validate()
    tw = M.twin_config(base, json.loads(CONFIG.read_text())["params"]["twin_overrides"])
    tw.validate()
    assert C1.twin_thresholds_ok(tw, base)
    t = C1.twin_thresholds(tw)
    assert t["p_ins_q32"] == t["p_neg_q32"] == t["inv_s_q32"] == 0 and t["s_num"] == 1 << 62
    diff = {k for k, v in tw.to_dict().items() if base.to_dict()[k] != v}
    assert diff == {"p_ins", "p_neg", "s"}
    r1 = json.loads((ROOT / "experiments" / "cvtm" / "configs" / "ssl_r1_controls.json").read_text())["params"]["twin_overrides"]
    ours = json.loads(CONFIG.read_text())["params"]["twin_overrides"]
    assert {k: ours[k] for k in ("p_ins", "p_neg", "s_log2")} == {k: r1[k] for k in ("p_ins", "p_neg", "s_log2")}


# ====================================================================== (d) slot transitions
def test_slot_transitions_counts_crossings_turnover_and_lowering():
    E = LIT_EMPTY
    lid0 = np.array([[5, 6, 7, E], [8, 9, E, E], [11, E, E, E]], np.uint32)
    st0 = np.array([[127, 130, 50, 0], [136, 140, 0, 0], [129, 0, 0, 0]], np.uint8)
    b0 = np.array([3, 4, 5], np.uint32)
    lid1 = np.array([[5, 6, 7, 12], [8, 9, E, E], [11, E, E, E]], np.uint32)
    st1 = np.array([[128, 127, 52, 123], [136, 140, 0, 0], [136, 0, 0, 0]], np.uint8)
    b1 = np.array([3, 4, 9], np.uint32)          # clause 2 recycled and reseeded with the same literal id
    t = M.slot_transitions(np, (lid0, st0, b0), (lid1, st1, b1))
    assert t == {"retained_changed": 3, "unit_steps_lb": 6, "cross_up": 1, "cross_down": 1, "lowered": 1,
                 "out": 1, "out_included": 1, "in_": 2, "in_included": 1}
    s = M.transition_summary(t)
    assert s["cross"] == 2 and s["frac_retained_changes_crossing"] == 2 / 3 and s["frac_unit_steps_crossing_ub"] == 2 / 6
    assert M.slot_transitions(np, (lid0, st0, b0), (lid0, st0, b0))["retained_changed"] == 0
    assert INCLUDE_THRESHOLD == 128


# ====================================================================== (e) random bank at positions, image codes
def _image_corpus(n_img=6, V=40, seed=3):
    rng = np.random.default_rng(seed)
    p = 1.0 / np.arange(1, V) ** 1.1
    p /= p.sum()
    tok = rng.choice(V - 1, size=(n_img, PT.GRID, PT.GRID), p=p).astype(np.uint16)
    tok = np.concatenate([tok, np.full((n_img, PT.GRID, 1), V - 1, np.uint16)], axis=2).reshape(-1)  # ROW_END = V - 1
    ds = PT.doc_starts(n_img)
    return tok, ds, make_kind_map(V, rng), V


def test_random_bank_at_every_position_is_the_confirm_random_bank():
    tok, ds, km, V = _image_corpus()
    a = M.random_bank_at(tok, ds, km, np.arange(tok.shape[0]), V, 500, seed=7)
    b = CF.random_bank(tok, ds, km, "masked", V, 500, seed=7)
    assert np.array_equal(a.col, b.col) and np.array_equal(a.sym, b.sym) and np.array_equal(a.n_lit, b.n_lit)
    assert a.info["redraws"] == b.info["redraws"]


def test_random_bank_at_draws_only_literals_true_at_the_given_positions():
    tok, ds, km, V = _image_corpus(n_img=8)
    pos = PT.patch_positions(np.array([1, 4])).astype(np.int64)
    bank = M.random_bank_at(tok, ds, km, pos, V, 400, seed=2)
    off, kind = CF._col_tables("masked")
    from tmlm.ssl import controls as K

    lo, hi = K.doc_bounds(np, ds)
    truth = {c: set(CF.symbol_at(tok, lo, hi, km, pos, np.full(pos.shape, off[c]), np.full(pos.shape, kind[c]), V).tolist())
             for c in range(off.shape[0])}
    for j in range(bank.n):
        for t in range(int(bank.n_lit[j])):
            assert int(bank.sym[j, t]) in truth[int(bank.col[j, t])]
    other = M.random_bank_at(tok, ds, km, pos, V, 400, seed=3)
    assert not np.array_equal(other.sym, bank.sym)


def test_randbank_codes_equal_the_reference_with_keep_and_df_rows():
    tok, ds, km, V = _image_corpus(n_img=7, seed=5)
    n_img = 7
    bank = M.random_bank_at(tok, ds, km, PT.patch_positions(np.arange(n_img)), V, 60, seed=1)
    single = CF.literal_single_counts(np, tok, ds, km, "masked", V)
    firer = CF.RandomBankFirer(np, bank, single, km)
    keep = np.array([1, 0, 1, 1, 0, 0, 1], bool)
    dfr = np.array([0, 1, 1, 0, 1, 1, 1], bool)
    sd, info, df = M.randbank_codes(np, firer, tok, ds, n_img, 3, keep=keep, df_rows=dfr)
    ref = M.randbank_reference_codes(bank, tok, ds, km, np.arange(n_img))
    want = ref.rows(np.flatnonzero(keep))
    assert np.array_equal(sd.indptr, want.indptr) and np.array_equal(sd.ids, want.ids.astype(np.uint32))
    assert np.array_equal(sd.cnt.astype(np.int64), want.cnt.astype(np.int64)) and sd.cnt.dtype == np.uint16
    dref = np.zeros(4 * bank.n, np.int64)
    for i in np.flatnonzero(dfr):
        r = ref.rows(np.asarray([i]))
        dref[r.ids.astype(np.int64)] += 1
    assert np.array_equal(df, dref)
    assert info["positions"] == n_img * 49 and sd.n_cols == 4 * bank.n
    one, _, _ = M.randbank_codes(np, firer, tok, ds, n_img, 100)
    assert np.array_equal(one.rows(np.flatnonzero(keep)).ids, sd.ids)                       # chunking does not matter


# ====================================================================== (f) diagnostics
def test_owned_per_class_and_class_sum_summary():
    st = {"alive": np.array([1, 1, 0, 1], np.uint8),
          "vote_cls": np.array([[0, 2, 0xFFFF], [2, 1, 0xFFFF], [0, 0xFFFF, 0xFFFF], [2, 0xFFFF, 0xFFFF]], np.uint16),
          "vote_w": np.array([[3, -1, 0], [1, 2, 0], [5, 0, 0], [-2, 0, 0]], np.int16)}
    o = M.owned_per_class(st, 3, 2)
    assert o["total_ownerships"] == 3 and o["classes_at_least_T"] == 0 and o["classes_below_T"] == 3   # dead clause 2 ignored
    assert o["quantiles"]["0.0"] == o["quantiles"]["1.0"] == 1.0 and o["classes_owned_by_none"] == 0
    assert M.owned_per_class(st, 3, 1)["classes_at_least_T"] == 3
    g = np.array([-70, 0, 3, 64, 200], np.int64)
    s = M.class_sum_summary(g, g + 1, g - 5, 64, max_log2=10)
    assert sum(s["gold"]["hist"]) == 5 and s["gold"]["frac_ge_T"] == 0.4 and s["gold"]["frac_le_minus_T"] == 0.2
    assert s["margin"]["quantiles"]["0.5"] == 5.0 and s["n_positions"] == 5
    e = M.class_sum_edges(3)
    assert np.histogram([-1.0, -0.5, 0.0, 0.5, 1.0, 8.0, 9.0], bins=e)[0].tolist() == [0, 0, 0, 0, 2, 2, 1, 0, 0, 2]


# ====================================================================== (g) report statistics and the label
def _acc(main_pts, rung_pts, twin_pts, seeds=(1, 2, 3), draws=5, budget=10, noise=0.2):
    rng = np.random.default_rng(0)
    acc = {}
    for s in seeds:
        for d in range(draws):
            base = 0.6 + 0.01 * rng.standard_normal()
            acc[("code_main", s, budget, d)] = base + main_pts / 100
            for i, r in enumerate(["ladder.c0y", "ladder.pct5", "ladder.pct10", "ladder.epoch1"]):
                acc[(r, s, budget, d)] = base + (rung_pts - i * 0.5) / 100 + noise / 100 * rng.standard_normal()
            acc[("twin", s, budget, d)] = base + twin_pts / 100 + noise / 100 * rng.standard_normal()
    return acc


@pytest.mark.parametrize("main,rung,twin,label", [(5.0, 1.0, 1.5, "literal-learning credit"),
                                                  (1.2, 1.0, 0.0, "no literal-learning credit"),
                                                  (5.0, 1.0, 4.5, "no literal-learning credit")])
def test_credit_label_needs_both_margins_with_the_interval_above_0(main, rung, twin, label):
    acc = _acc(main, rung, twin)
    rungs = ["ladder.c0y", "ladder.pct5", "ladder.pct10", "ladder.epoch1"]
    mx = M.max_ladder(acc, rungs, 10)
    assert set(mx.values()) == {"ladder.c0y"}
    lad = M.row_stats(M.diff_units(acc, "code_main", mx, 10), 2000, 1)
    tw = M.row_stats(M.diff_units(acc, "code_main", "twin", 10), 2000, 1)
    assert lad["pairs"] == 15 and set(lad["per_seed_mean"]) == {"1", "2", "3"}
    assert abs(lad["two_level_bootstrap"]["mean"] - (main - rung)) < 0.5
    assert M.credit_label(lad, tw, 3, 1.0)["label"] == label
    assert M.credit_label(lad, tw, 4, 1.0)["label"] == CF.INCOMPLETE           # a missing seed


def _write_run(d: Path, metrics: dict, cells: list[dict] | None = None) -> Path:
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(json.dumps({k: {"value": v, "label": "measured"} for k, v in metrics.items()}))
    if cells is not None:
        (d / "cells.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cells))
    return d


def test_report_end_to_end_on_synthetic_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("SSL_SCRATCH", str(tmp_path / "scratch"))
    (tmp_path / "scratch").mkdir()
    M._SCRATCH.clear()
    seed, budgets, draws = 100, (10, 100), 2
    reg, ctl = [], []
    for b in budgets:
        for d in range(draws):
            lab = f"lab{b}{d}"
            for rep, a in (("code_main", 0.70), ("code_young", 0.60), ("bovw", 0.55)):
                reg.append({"seed": seed, "budget": b, "draw": d, "rep": rep, "probe": "lr", "acc": a + 0.001 * d,
                            "labelled_sha256": lab, "pred_sha256": f"p{rep}{b}{d}"})
            for rep, a in (("ladder.c0y", 0.60), ("ladder.pct5", 0.62), ("ladder.pct10", 0.63), ("ladder.epoch1", 0.66),
                           ("twin", 0.64), ("randbank", 0.50)):
                ctl.append({"seed": seed, "budget": b, "draw": d, "rep": rep, "probe": "lr", "acc": a,
                            "labelled_sha256": lab, "source": M.SOURCE,
                            "pred_sha256": f"pcode_young{b}{d}" if rep == "ladder.c0y" else f"x{rep}"})
    runs = [
        _write_run(tmp_path / "gpu", {"stage": "gpu", "seed": seed, "main.epoch1": {"dev_masked_top1": 0.5}, "main.young.dev_masked_top1": 0.3}),
        _write_run(tmp_path / "probe", {"stage": "probe", "seed": seed}, reg),
        _write_run(tmp_path / "cprobe", {"controls.stage": "probe", "seed": seed}, ctl),
        _write_run(tmp_path / "replay", {"controls.stage": "replay", "seed": seed,
                                         "ladder.rungs": {"c0y": {"dev_masked_top1": 0.3}, "epoch1": {"dev_masked_top1": 0.5}}}),
    ]
    rc = M.main(["--stage", "report", "--config", str(SMOKE), "--runs", ",".join(str(r) for r in runs)])
    assert rc == 0
    rep = json.loads((tmp_path / "scratch").glob("ssl_r4c_report_*").__next__().joinpath("report.json").read_text())
    row = rep["rows"]["b10.minus_max_ladder"]
    assert row["reference"] == {"100": "ladder.epoch1"}
    assert abs(row["two_level_bootstrap"]["mean"] - (70.05 - 66.0)) < 1e-9
    assert rep["labels"]["10"]["label"] == "literal-learning credit"
    assert abs(rep["rows"]["b100.minus_randbank"]["per_seed_mean"]["100"] - 20.05) < 1e-9
    # a control cell with a different labelled sample, or a C0y prediction unlike code_young, fails the report
    M._SCRATCH.clear()
    bad = [dict(c) for c in ctl]
    bad[0]["labelled_sha256"] = "other"
    runs[2] = _write_run(tmp_path / "cprobe_bad", {"controls.stage": "probe", "seed": seed}, bad)
    assert M.main(["--stage", "report", "--config", str(SMOKE), "--runs", ",".join(str(r) for r in runs)]) == 1


# ====================================================================== (h) configs
@pytest.mark.parametrize("path", [CONFIG, SMOKE], ids=["confirmatory", "smoke"])
def test_configs_are_complete_and_read_every_r4_parameter_from_the_r4_config(path):
    cj, r4 = M.load_config(path)
    p = cj["params"]
    assert cj["script"] == "experiments/ssl_r4_controls.py" and (ROOT / cj["script"]).exists()
    assert cj["prereg"] == []
    for k in ("young_ladder", "twin_overrides", "random_bank", "diag", "report", "seeds"):
        assert k in p
    for k in ("epochs", "young_fraction", "engine", "probe", "f1", "n_codes"):
        assert k not in p and k in r4["params"]                     # never copied, always read from r4_config
    EngineConfig(**{**r4["params"]["engine"], "mode": "masked", "vocab": 2000}).validate()
    assert p["report"]["label_budget"] == r4["params"]["verdict"]["budget"] == 10
    assert p["report"]["label_threshold_points"] == 1.0 and p["report"]["budgets"] == r4["params"]["probe"]["budgets"]
    text = path.read_text(encoding="utf-8")
    assert "mem_limits_gib" in cj


def test_confirmatory_config_states_its_timing_f29_and_cost():
    cj, r4 = M.load_config(CONFIG)
    assert cj["r4_config"] == "experiments/cvtm/configs/ssl_r4_images.json" and cj["params"]["seeds"] == [1, 2, 3]
    assert cj["params"]["smoke"] is False and cj["params"]["report"]["n_boot"] == 10000
    assert "BEFORE any confirmatory R4 output" in cj["status"] and "F29" in cj["f29"] and "F29" in cj["docs"]
    assert set(cj["mem_limits_gib"]) >= {"replay", "twin", "randbank", "probe", "diag", "report"}
    ce = cj["cost_estimate"]
    assert ce["label"].startswith("derived") and "seed100_exploratory" in ce["sources"]["gpu"] and "seed100_exploratory" in ce["sources"]["probe"]
    for src in ce["sources"].values():
        run = ROOT / src.split(":")[0]
        assert run.exists() and json.loads((run / "run.json").read_text())["wall_time_s"] > 0
    smoke, _ = M.load_config(SMOKE)
    assert smoke["params"]["smoke"] is True and smoke["params"]["seeds"] == [100]
    assert smoke["r4_config"] == "experiments/cvtm/configs/ssl_r4_images_smoke.json"
