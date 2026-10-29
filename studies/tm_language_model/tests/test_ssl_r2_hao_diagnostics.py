"""Tests of the H-AO hierarchy diagnostics (docs/13 A8 item 3): src/tmlm/ssl/hao_diag.py and
experiments/ssl_r2_hao_diagnostics.py.

CPU (deterministic): Valiant elimination recovers planted OR-groups on a tiny world and predicts unseen combinations;
the TA-state bit-plane encoding round-trips and agrees with the top-plane decoding of ssl_r2_hao.pkg_included; the
oracle hierarchy predicts the concept exactly under the numpy reference of the package's inference (both negative
clause designs, with and without binary inference); node recovery; the set_state audit on BOTH package sources
(c85ca4e7 installed in .venv-tm, 5b4a223 from third_party/ref); the MultiConceptDemo port; the config; the
tuning-reproduction check; the valiant, controls and report stages end to end on a tiny world.

GPU (marked gpu; skipped when pycuda cannot create a context): the oracle write round-trips on tiny package instances
(full and partly filled last literal chunk, depths deep and boundary, both literal sets: raw words, weights,
includes, the package's get_ta_states kernel, class sums equal to the reference, exact predictions, fit() keeps the
written state), a smoke run of every package stage on a tiny world, and the selftest stage. Run through the locks,
CPU slot first (docs/12 F25):
    scripts/cpu_slot.sh 1 -- scripts/gpu_run.sh shared -- .venv-tm/bin/python -m pytest -q tests/test_ssl_r2_hao_diagnostics.py
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tmlm.ssl import hao_diag as HG
from tmlm.ssl import hao_world as H

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import ssl_r2_hao as E  # noqa: E402  (plain imports: a broken import of the module under test fails loudly)
import ssl_r2_hao_diagnostics as DG  # noqa: E402

REF = ROOT / "third_party" / "ref" / "PyHierarchicalTsetlinMachineCUDA"
CONFIG = ROOT / "experiments" / "cvtm" / "configs" / "ssl_r2_hao_diagnostics.json"


def _tiny(seed=3, vocab=200, n_groups=3, group_size=4, n_distractors=6, extra=(0.5, 0.3, 0.2), n=1600, train_fraction=0.6):
    w = H.make_world(seed, vocab=vocab, n_groups=n_groups, group_size=group_size, train_fraction=train_fraction)
    fit_mask, val_mask = H.carve_combos(w, w.train_combo, 0.2, 5)
    kw = {"n_distractors": n_distractors, "extra_probs": extra}
    return w, H.sample_bags(w, n, fit_mask, 3, **kw), H.sample_bags(w, 400, val_mask, 4, **kw)


@pytest.fixture
def fresh_runner(tmp_path, monkeypatch):
    monkeypatch.setenv("HAO_DIAG_OUT", str(tmp_path / "out"))
    DG._OUT.clear()
    DG.METRICS.clear()
    DG.FAILED.clear()
    yield tmp_path / "out"
    DG._OUT.clear()


# ------------------------------------------------------------------ Valiant elimination
def test_valiant_recovers_planted_groups_on_a_tiny_world():
    w, fit, unseen = _tiny(train_fraction=0.9)  # every word in several fit combinations
    model = HG.valiant_fit(fit.indptr, fit.indices, fit.y, w.vocab, min_freq=0.1, max_or=4)
    truth = sorted(tuple(int(x) for x in g) for g in w.groups)
    assert sorted(model["kept"]) == sorted(int(x) for x in w.groups.reshape(-1))  # distractors filtered (6/188 < 0.1)
    assert sorted(model["minimal"]) == truth
    for bags in (fit, unseen):
        p = HG.valiant_predict(model, bags.indptr, bags.indices)
        assert np.array_equal(p, bags.y.astype(bool))
    # unseen positives hold only combinations absent from every fit bag: they are still accepted
    fit_combos = {c for i in np.flatnonzero(fit.y == 1) for c in H.contained_combos(w, fit.bag(i))}
    un_combos = {c for i in np.flatnonzero(unseen.y == 1) for c in H.contained_combos(w, unseen.bag(i))}
    assert un_combos and not (un_combos & fit_combos)


def test_valiant_with_few_positives_keeps_spurious_ors_and_minimal_predicts_like_all():
    w, fit, unseen = _tiny(seed=4)
    model = HG.valiant_fit(fit.indptr, fit.indices, fit.y, w.vocab, min_freq=0.1, max_or=4, n_pos=6)
    assert model["n_pos"] == 6
    assert model["n_survivors"] >= len(model["minimal"]) >= 1
    for bags in (fit, unseen):
        assert np.array_equal(HG.valiant_predict(model, bags.indptr, bags.indices, "minimal"),
                              HG.valiant_predict(model, bags.indptr, bags.indices, "all"))
    # survivors are exactly the candidates hit by every used positive bag (brute force)
    kept = model["kept"]
    pos = np.flatnonzero(fit.y == 1)[:6]
    brute = 0
    for r in range(1, 5):
        for comb in itertools.combinations(kept, r):
            if all(set(comb) & set(fit.bag(i).tolist()) for i in pos):
                brute += 1
    assert brute == model["n_survivors"]


def test_valiant_refuses_too_many_candidates():
    w, fit, _ = _tiny()
    with pytest.raises(ValueError, match="max_candidates"):
        HG.valiant_fit(fit.indptr, fit.indices, fit.y, w.vocab, min_freq=0.1, max_or=4, max_candidates=10)


# ------------------------------------------------------------------ TA state encoding
@pytest.mark.parametrize("L", [31, 32, 33, 4000])
def test_ta_state_encoding_round_trips_and_matches_top_plane_decoding(L):
    rng = np.random.default_rng(L)
    st = rng.integers(0, 256, size=(3, 5, L)).astype(np.uint32)
    raw = HG.encode_ta_state(st, 8)
    assert raw.dtype == np.uint32 and raw.size == 3 * 5 * HG.n_chunks(L) * 8
    assert np.array_equal(HG.decode_ta_state(raw, 3, 5, L, 8), st)
    assert np.array_equal(HG.include_from_raw(raw, 3, 5, L, 8), st >= 128)
    # the package's layout: bit (k mod 32) of word [clause][team][k // 32][plane]
    r4 = raw.reshape(3, 5, HG.n_chunks(L), 8)
    for (c, t, k) in [(0, 0, 0), (2, 4, L - 1), (1, 2, L // 2)]:
        v = sum(((int(r4[c, t, k // 32, b]) >> (k % 32)) & 1) << b for b in range(8))
        assert v == int(st[c, t, k])
    # padding TAs carry the package's initial value 2^(bits-1) - 1
    if L % 32:
        pad = HG.decode_ta_state(raw, 3, 5, HG.n_chunks(L) * 32, 8)[:, :, L:]
        assert np.all(pad == 127)


# ------------------------------------------------------------------ oracle hierarchy and the reference inference
@pytest.mark.parametrize("neg", [True, False])
@pytest.mark.parametrize("binary", [False, True])
def test_oracle_reference_predicts_the_concept(neg, binary):
    w, fit, unseen = _tiny(seed=5, extra=(1.0,), n=400)
    orc = HG.oracle_hierarchy(w.groups, w.vocab, clauses=8, alternatives=5, append_negated=neg)
    assert orc.design == ("de_morgan" if neg else "bias")
    assert orc.inc.shape == (8, 5 * 3, 2 * w.vocab if neg else w.vocab)
    assert np.array_equal(orc.weights, np.array([1, -1] * 4, np.int32))
    for bags in (fit, unseen):
        s, out = HG.reference_scores(orc.inc, orc.weights, bags.X, 5, 3, w.vocab, neg, binary)
        assert np.array_equal(s >= 0, bags.y.astype(bool))
        roles = [r for r, _ in orc.roles]
        if not neg:
            assert np.all(out[:, roles.index("bias")] == 1)
            assert np.all(out[:, [i for i, r in enumerate(roles) if r == "never"]] == 0)
        # positive clauses fire exactly on positives, with the product of per-group counts
        assert np.array_equal(out[:, 0] > 0, bags.y.astype(bool))
    # the concept groups the existing read-out finds in the positive clauses are the true groups
    groups = H.concept_groups(orc.inc, 5, 3, neg)
    assert all(g == [set(int(x) for x in t) for t in w.groups] for g in groups[0::2])


def test_leaf_outputs_against_a_direct_loop():
    rng = np.random.default_rng(0)
    V, C, J = 12, 3, 4
    inc = rng.random((C, J, 2 * V)) < 0.12
    inc[0, 0] = False  # an empty leaf is true
    X = (rng.random((50, V)) < 0.4).astype(np.uint8)
    got = HG.leaf_outputs(inc, X, V, True)
    lit = np.concatenate([X, 1 - X], axis=1).astype(bool)
    want = np.array([[np.all(lit[n][inc[c, j]]) for c in range(C) for j in range(J)] for n in range(50)])
    assert np.array_equal(got, want)


def test_oracle_rejects_too_few_alternatives_and_de_morgan_without_negation():
    w = H.make_world(1)
    with pytest.raises(ValueError, match="alternatives"):
        HG.oracle_hierarchy(w.groups, w.vocab, 4, 5, True)
    with pytest.raises(ValueError, match="negated"):
        HG.oracle_hierarchy(w.groups, w.vocab, 4, 8, False, design="de_morgan")
    with pytest.raises(ValueError, match="one negative clause per group"):
        HG.oracle_hierarchy(w.groups, w.vocab, 6, 8, True)  # 3 negative clauses for 4 groups


def test_oracle_states_depths():
    inc = np.array([[[True, False]]])
    assert HG.oracle_states(inc, 8, "deep").tolist() == [[[255, 0]]]
    assert HG.oracle_states(inc, 8, "boundary").tolist() == [[[128, 127]]]


def test_node_recovery_is_perfect_for_the_oracle_and_sees_damage():
    w = H.make_world(2, vocab=100, n_groups=3, group_size=4, train_fraction=0.6)
    orc = HG.oracle_hierarchy(w.groups, w.vocab, 6, 4, True)
    r = HG.node_recovery(orc.inc, orc)
    assert r["pos_node_jaccard_mean"] == 1.0 and r["dm_node_jaccard_mean"] == 1.0 and r["clauses_exact_share"] == 1.0
    bad = orc.inc.copy()
    g0 = int(w.groups[0, 0])
    bad[0, [j for j in range(12) if (j // 4) % 3 == 0 and j % 4 == 0], g0] = False  # drop word 0 of group 0 in node 0 of clause 0
    bad[0, 1, int(w.distractors[0])] = True  # add a distractor to node 0 of clause 0
    r2 = HG.node_recovery(bad, orc)
    assert r2["pos_node_jaccard_min"] == pytest.approx(3 / 5)
    assert r2["literals_added"]["positive"] == 1 and r2["literals_removed"]["positive"] == 1
    assert r2["clauses_exact_share"] == pytest.approx(5 / 6)


# ------------------------------------------------------------------ set_state audit at both commits (source only)
def _installed_tm_source() -> str | None:
    spec = importlib.util.find_spec("PyHierarchicalTsetlinMachineCUDA")  # the package __init__ is empty: no CUDA context
    if spec is None or not spec.submodule_search_locations:
        return None
    return (Path(list(spec.submodule_search_locations)[0]) / "tm.py").read_text(encoding="utf-8")


def _ref_source(commit: str, path: str) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(REF), "show", f"{commit}:{path}"], capture_output=True, text=True,
                              check=True, timeout=60).stdout
    except (subprocess.SubprocessError, OSError):
        return None


@pytest.mark.parametrize("which", ["installed", DG.C85, DG.HD.PIN])
def test_set_state_under_allocates_by_the_chunks_per_leaf_factor(which):
    src = _installed_tm_source() if which == "installed" else _ref_source(which, "PyHierarchicalTsetlinMachineCUDA/tm.py")
    if src is None:
        pytest.skip(f"package source {which} not available")
    for chunks in (125, 63):  # 2,000 features with and without negated literals
        attrs = {"number_of_clauses": 64, "number_of_ta_teams": 32, "number_of_literal_chunks_per_leaf": chunks,
                 "number_of_state_bits": 8, "number_of_outputs": 1}
        a = HG.set_state_allocation(src, attrs)
        assert a["under_allocates"] and a["needed_over_alloc"] == chunks
        assert a["alloc_bytes"] == 64 * 32 * 8 * 4
        assert a["sets_first"] is False


# ------------------------------------------------------------------ demo ports and config
def test_multiconcept_data_is_seeded_and_follows_the_rule():
    a = HG.multiconcept_data(7, 16, 300, 0.0)
    b = HG.multiconcept_data(7, 16, 300, 0.0)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    Xtr, ytr, ytr_clean, Xte, yte = a
    assert Xtr.shape == (300, 32) and np.all(Xtr.sum(axis=1) == 2) and np.all(Xtr[:, :16].sum(axis=1) == 1)
    assert np.array_equal(HG.multiconcept_rule(Xtr, 16), ytr_clean) and np.array_equal(HG.multiconcept_rule(Xte, 16), yte)
    assert np.array_equal(ytr, ytr_clean)  # noise 0.0
    noisy = HG.multiconcept_data(7, 16, 2000, 0.2)
    assert 0.15 < np.mean(noisy[1] != noisy[2]) < 0.25


def test_parse_argparse_defaults():
    src = ('import argparse\np = argparse.ArgumentParser()\np.add_argument("--epochs", default=10, type=int)\n'
           'p.add_argument("--s", default=2.5, type=float)\np.add_argument("--vanilla", action="store_true")\n'
           'p.add_argument("--number-of-ands", default=4, type=int)\n')
    assert HG.parse_argparse_defaults(src) == {"epochs": 10, "s": 2.5, "number_of_ands": 4}


def test_config_is_consistent():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    P = cfg["params"]
    assert cfg["script"] == "experiments/ssl_r2_hao_diagnostics.py" and cfg["prereg"] == []
    assert P["world_seeds"] == [100, 101]
    # the fresh unseen stream is new: not a stream of ssl_r2_hao.py and not the seen stream
    assert P["streams"]["unseen_fresh"] == 22 and P["n_unseen_fresh"] == 4000
    assert P["streams"]["unseen_fresh"] not in set(E.STREAM.values()) | {P["streams"]["seen"]}
    R = cfg["reading_notes"]
    assert "EXACTLY the validation bags" in R["seen_unseen"]["seed100_bias"] and "101" in R["seen_unseen"]["seed100_bias"]
    assert "POSITIVE literals only" in R["oracle"]["node_jaccard_note"] and "literals_added" in R["oracle"]["node_jaccard_note"]
    assert "load(state_dict)" in P["oracle"]["write"] and "out-of-bounds device write" in P["oracle"]["write"]
    Q = P["controls"]
    assert Q["models"] == ["tmu_tm", "tmu_ctm", "lr", "gbt", "mlp"] and Q["tuned_config"] in P["tuned_configs"]
    for tag, rel in P["tuned_configs"].items():
        T = json.loads((ROOT / rel).read_text(encoding="utf-8"))
        assert all(T["params"]["world"][k] == v for k, v in P["world"].items()), tag
        for stage in ("seen_unseen", "pospair", "oracle", "controls"):
            for m in P[stage]["models"]:
                assert m in T["params"]["models"]
        for m, point in Q["grid_choice"].items():
            assert point in T["params"]["models"][m]["grid"] and m in Q["grid_choice_val_acc_seed100"]
        for m, ch in T["tuning"]["chosen"].items():  # the tuned epoch count is the chosen epoch (F1 reproduces it)
            assert T["params"]["models"][m]["epochs"] == ch["chosen_epoch"], (tag, m)
        for m in P["oracle"]["models"]:
            spec = T["params"]["models"][m]
            assert spec["kind"] == "hier" and spec["alternatives"] >= P["world"]["group_size"]
            assert spec["n_concepts"] == P["world"]["n_groups"]
        for m in P["pospair"]["models"]:
            assert T["params"]["models"][m]["append_negated"] is False
    sel = DG.best_hier_config(cfg)["best"]
    assert (sel["tag"], sel["model"]) == ("head", "htm_hier_neg")
    for tag, demos in P["demos"].items():
        commit = DG.C85 if tag == "c85" else DG.HD.PIN
        for name, C in demos.items():
            src = _ref_source(commit, C["source"])
            if src is None:
                pytest.skip("reference clone not readable")
            assert DG.hashing.sha256_bytes(src.encode("utf-8")) == C["source_sha256"]
            assert HG.parse_argparse_defaults(src) == C["defaults"]
            assert set(C["variants"]) <= set(C["machine_clauses"])


# ------------------------------------------------------------------ tuning reproduction (F1)
def test_tuning_reproduction_compares_correct_bag_counts():
    choice = {"train_acc_at_chosen_epoch": 0.902, "val_acc_at_chosen_epoch": 0.75, "chosen_epoch": 60, "tuning_config": "h25"}
    row = {"epoch": 60, "fit_acc": 7216 / 8000, "unseen_acc": 3000 / 4000}
    ok, d = DG.tuning_reproduction(row, choice, 8000, 4000)
    assert ok and d["correct_bags"] == d["correct_bags_tuning"] == {"fit": 7216, "unseen": 3000}
    assert not DG.tuning_reproduction({**row, "unseen_acc": 2999 / 4000}, choice, 8000, 4000)[0]   # one bag off
    assert not DG.tuning_reproduction({**row, "fit_acc": 7217 / 8000}, choice, 8000, 4000)[0]


def test_check_tuning_reproduction_only_at_the_reference_seed_and_the_tuned_epoch(fresh_runner):
    cfg = {"params": {"reference": {"seed": 100}}}
    S = {"fit": SimpleNamespace(n=8000), "unseen": SimpleNamespace(n=4000)}
    choice = {"train_acc_at_chosen_epoch": 0.902, "val_acc_at_chosen_epoch": 0.75, "chosen_epoch": 60}
    good = {"epoch": 60, "fit_acc": 7216 / 8000, "unseen_acc": 3000 / 4000}
    cell = {"curve": [{"epoch": 10, "fit_acc": 0.6, "unseen_acc": 0.5}, good], "epochs_run": 60, "stop_reason": "epochs"}
    DG.check_tuning_reproduction(cfg, "a", 101, cell, choice, S)           # seed 101: not checked, reason recorded
    assert "check.a.reproduces_tuning_run" not in DG.METRICS and "101" in DG.METRICS["a.reproduces_tuning_run"]["reason"]
    DG.check_tuning_reproduction(cfg, "b", 100, {**cell, "curve": cell["curve"][:1]}, choice, S)  # epoch 60 not evaluated
    assert "check.b.reproduces_tuning_run" not in DG.METRICS and "no evaluated epoch 60" in DG.METRICS["b.reproduces_tuning_run"]["reason"]
    DG.check_tuning_reproduction(cfg, "c", 100, cell, choice, S)
    assert DG.METRICS["check.c.reproduces_tuning_run"]["value"] is True and not DG.FAILED
    bad = {**cell, "curve": [{**good, "unseen_acc": 0.7}]}
    DG.check_tuning_reproduction(cfg, "d", 100, bad, choice, S)
    assert DG.FAILED == ["d.reproduces_tuning_run"]


# ------------------------------------------------------------------ CPU stages end to end on a tiny world
def _tiny_cfg(tmp_path: Path, tag: str = "c85") -> Path:
    return DG.tiny_config(json.loads(CONFIG.read_text(encoding="utf-8")), tag, tmp_path)


def _rows(out: Path, stage: str) -> list[dict]:
    return [r for r in (json.loads(x) for x in (out / "cells.jsonl").read_text().splitlines() if x.strip()) if r["stage"] == stage]


def test_valiant_and_report_stages_on_a_tiny_world(tmp_path, fresh_runner):
    cfg = _tiny_cfg(tmp_path)
    assert DG.main(["--stage", "valiant", "--config", str(cfg), "--world-seed", "100"]) == 0, DG.FAILED
    rows = _rows(fresh_runner, "valiant")
    assert [r["n_pos_arg"] for r in rows] == [10, "all"]
    top = rows[-1]
    assert top["n_true_groups_recovered"] == 3 and top["unseen_acc"] == 1.0 and top["unseen_neg_acc"] == 1.0
    assert top["unseen_fresh_acc"] == 1.0 and DG.METRICS["valiant.seed100.nall.unseen_fresh_acc"]["value"] == 1.0
    DG.METRICS.clear()
    assert DG.main(["--stage", "report", "--config", str(cfg), "--runs", str(fresh_runner)]) == 0
    rd = json.loads((fresh_runner / "readings.json").read_text())
    assert rd["valiant"]["seed100"]["learns_the_concept"] is True
    assert rd["oracle_factorial_trigger"]["triggered"] is False


def test_world_sets_add_fresh_unseen_bags_of_the_validation_combinations(tmp_path, fresh_runner):
    cfg = json.loads(_tiny_cfg(tmp_path).read_text(encoding="utf-8"))
    S = DG.world_sets(cfg, 100)
    assert not DG.FAILED, DG.FAILED
    assert S["unseen_fresh"].n == 120 and not np.array_equal(S["unseen_fresh"].X, S["unseen"].X[:120])
    for name, mask in (("unseen", S["val_mask"]), ("unseen_fresh", S["val_mask"]), ("seen", S["fit_mask"])):
        assert all(H.check_bags(S["world"], S[name], mask, DG.TINY_WORLD["n_distractors"]).values()), name
    assert DG.METRICS["seed100.data.unseen_is_tuning_validation"]["value"] is False   # tiny config: reference seed 0


def test_controls_stage_on_a_tiny_world(tmp_path, fresh_runner):
    cfg = _tiny_cfg(tmp_path)
    assert DG.main(["--stage", "controls", "--config", str(cfg), "--world-seed", "100"]) == 0, DG.FAILED
    rows = _rows(fresh_runner, "controls")
    assert [r["model"] for r in rows] == ["tmu_tm", "tmu_ctm", "lr", "gbt", "mlp"]
    for r in rows:
        f = r["final"]
        for n in DG.SETS:
            assert 0.0 <= f[f"{n}_acc"] <= 1.0
        assert f["gap_seen_minus_unseen_fresh"] == f["seen_acc"] - f["unseen_fresh_acc"]
    assert rows[0]["params"]["s"] == 20.0 and rows[1]["params"]["s"] == 5.0    # the grid points of params.controls
    DG.METRICS.clear()
    assert DG.main(["--stage", "report", "--config", str(cfg), "--runs", str(fresh_runner)]) == 0
    rd = json.loads((fresh_runner / "readings.json").read_text())
    assert set(rd["controls"]) == {f"cpu.seed100.{m}" for m in ("tmu_tm", "tmu_ctm", "lr", "gbt", "mlp")}
    for v in rd["controls"].values():
        assert v["unseen_is_tuning_validation"] is False and "generalises_to_unseen_combinations_fresh" in v


def test_exploratory_seeds_only(tmp_path, fresh_runner):
    cfg = _tiny_cfg(tmp_path)
    with pytest.raises(SystemExit):
        DG.main(["--stage", "valiant", "--config", str(cfg), "--world-seed", "1"])


# ------------------------------------------------------------------ GPU
@pytest.fixture(scope="module")
def htm():
    try:
        import pycuda.driver as cuda

        cuda.init()
        if cuda.Device.count() < 1:
            raise RuntimeError("no CUDA device visible")
        import PyHierarchicalTsetlinMachineCUDA.tm as mod  # creates the primary context
    except Exception as exc:  # noqa: BLE001 - any failure here means: no usable GPU context
        pytest.skip(f"pycuda cannot create a CUDA context: {type(exc).__name__}: {exc}")
    return mod


def _tag(htm_mod) -> str:
    prov = E.pkg_provenance()
    return DG.COMMITS[((prov.get("direct_url") or {}).get("vcs_info") or {}).get("commit_id")]


@pytest.mark.gpu
@pytest.mark.parametrize("neg", [True, False])
@pytest.mark.parametrize("vocab,alternatives,depth", DG.ROUND_TRIP_CASES)
def test_oracle_write_round_trips_on_a_tiny_package_instance(htm, neg, vocab, alternatives, depth):
    """Write, read back (raw words, weights, includes, the package's get_ta_states), class sums equal to the numpy
    reference, exact predictions, and one fit() at s = 1e9 that keeps the written state. vocab 60 with 5 alternatives
    for 4-word groups: a partly filled last literal chunk (120 or 60 literals) and a spare alternative."""
    tag = _tag(htm)
    sw = {"binary_inference": 1, "constant_update_p": 1} if (tag == "head" and not neg) else {}
    r = DG.oracle_round_trip(htm, tag, vocab, alternatives, depth, neg, sw)
    assert all(r["checks"].values()), r
    info = r["info"]
    assert info["literals"] == (2 * vocab if neg else vocab) and info["chunks_per_leaf"] == HG.n_chunks(info["literals"])
    assert info["last_chunk_partly_filled"] == (vocab == 60)
    if depth == "deep":
        assert info["includes_added_by_fit"] == 0 and info["clauses_exact_share_after_fit"] == 1.0


@pytest.mark.gpu
def test_package_stages_smoke_on_a_tiny_world(htm, tmp_path, fresh_runner):
    tag = _tag(htm)
    cfg = _tiny_cfg(tmp_path, tag)
    for stage in ("oracle", "seen_unseen", "pospair", "threshold"):
        DG.METRICS.clear()
        rc = DG.main(["--stage", stage, "--config", str(cfg), "--world-seed", "100", "--replay-check"])
        assert rc == 0, (stage, DG.FAILED)
        if stage == "seen_unseen":  # tiny config: reference seed 0, so the tuning reproduction records why it is not checked
            for model in ("htm_hier_neg", "htm_flat_neg"):
                assert "is not the tuning seed" in DG.METRICS[f"seen_unseen.{tag}.seed100.{model}.reproduces_tuning_run"]["reason"]
    orc_rows = _rows(fresh_runner, "oracle")
    assert len(orc_rows) == 2 * 2  # 2 models x 2 depths x 1 s
    for r in orc_rows:
        assert r["verified"] and r["read_back"]["raw_words_equal"] and r["curve"][0]["epoch"] == 0
        assert r["curve"][0]["unseen_acc"] == 1.0 and r["curve"][0]["pos_node_jaccard_mean"] == 1.0
    assert {r["model"] for r in _rows(fresh_runner, "seen_unseen")} == {"htm_hier_neg", "htm_flat_neg"}
    assert {r["model"] for r in _rows(fresh_runner, "pospair")} == {"htm_hier_pos", "htm_flat_pos"}
    th = _rows(fresh_runner, "threshold")[0]
    assert th["model"] == "htm_hier_neg" and th["epochs_run"] <= 3
    DG.METRICS.clear()
    assert DG.main(["--stage", "report", "--config", str(cfg), "--runs", str(fresh_runner)]) == 0
    rd = json.loads((fresh_runner / "readings.json").read_text())
    assert set(rd["oracle"]) == {f"{tag}.htm_hier_neg.deep.s4", f"{tag}.htm_hier_neg.boundary.s4",
                                 f"{tag}.htm_hier_pos.deep.s4", f"{tag}.htm_hier_pos.boundary.s4"}


@pytest.mark.gpu
def test_selftest_stage_on_the_installed_commit(htm, fresh_runner):
    if _ref_source(DG.C85, "examples/ANDOfXORsDemo.py") is None:  # the demos stage reads the reference clone
        pytest.skip("reference clone not readable (third_party/ref/PyHierarchicalTsetlinMachineCUDA, not distributed)")
    tag = _tag(htm)
    other = "head" if tag == "c85" else "c85"
    assert DG.main(["--stage", "selftest", "--config", str(CONFIG), "--expect-tag", other]) == 1
    assert DG.FAILED == ["selftest.package_commit_is_expected"]
    DG.METRICS.clear()
    DG.FAILED.clear()
    assert DG.main(["--stage", "selftest", "--config", str(CONFIG), "--expect-tag", tag]) == 0, DG.FAILED
    assert DG.METRICS["selftest.passed"]["value"] is True
    names = [k for k in DG.METRICS if k.startswith("check.selftest.round_trip.")]
    assert len(names) == 2 * len(DG.ROUND_TRIP_CASES)
    assert not (fresh_runner / "cells.jsonl").exists()          # the tiny stages never write the run's own cells
    rows = _rows(fresh_runner / "selftest", "oracle")
    assert len(rows) == 4 and all(r["tag"] == tag for r in rows)
    with pytest.raises(SystemExit):
        DG.main(["--stage", "selftest", "--config", str(CONFIG)])  # --expect-tag is required


@pytest.mark.gpu
def test_demo_variant_runs_on_tiny_defaults(htm, fresh_runner):
    tag = _tag(htm)
    C = {"defaults": {"epochs": 2, "number_of_clauses": 2, "number_of_state_bits": 8, "number_of_irrelevant_features": 1,
                      "number_of_training_examples": 500, "number_of_testing_examples": 200, "T": 40, "s": 2.5,
                      "number_of_alternatives": 10, "number_of_ands": 4, "noise": 0.01},
         "runs": 2, "machine_clauses": {"hier": ["number_of_clauses"]}, "data_seed": 1, "machine_seed_seed": 2,
         "machine_seed_randint": [2147483647], "learn_threshold": 0.95, "wall_cap_s": 600}
    a = DG.run_demo_variant(htm, tag, "andxor", C, "hier")
    b = DG.run_demo_variant(htm, tag, "andxor", C, "hier")
    assert len(a["runs"]) == 2 and all(len(r["curve"]) == 2 for r in a["runs"])
    assert [[e["test_acc"] for e in r["curve"]] for r in a["runs"]] == [[e["test_acc"] for e in r["curve"]] for r in b["runs"]]
    assert [r["machine_seed"] for r in a["runs"]] == [r["machine_seed"] for r in b["runs"]]
    assert a["runs"][0]["machine_seed"] != a["runs"][1]["machine_seed"]
    Cm = {"defaults": {"epochs": 2, "number_of_clauses": 2, "number_of_examples": 300, "T": 128, "s": 21.1,
                       "number_of_alternatives": 8, "number_of_elements": 16, "number_of_copies": 2, "noise": 0.0},
          "runs": 1, "state_bits": 8, "machine_clauses": {"hier": ["number_of_clauses"]}, "data_seed": 3,
          "machine_seed_seed": 4, "machine_seed_randint": [10000], "learn_threshold": 0.95, "wall_cap_s": 600}
    m1 = DG.run_demo_variant(htm, tag, "multiconcept", Cm, "hier")
    m2 = DG.run_demo_variant(htm, tag, "multiconcept", Cm, "hier")
    assert [e["test_acc"] for e in m1["runs"][0]["curve"]] == [e["test_acc"] for e in m2["runs"][0]["curve"]]
    assert not DG.FAILED
