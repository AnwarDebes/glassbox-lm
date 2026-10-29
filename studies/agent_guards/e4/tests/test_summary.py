"""Tests of the registered statistic and the summary (scripts/summarize_e4.py): known inputs give known intervals, the
completeness rule, E4.5's not-informative rule, the two-level interval, and an END-TO-END run of summarize() on
stand-in data (the shipped gpt-4.1 retail trajectories laid out as arm folders with guard logs rebuilt from them).
Revision 3 (re-review item 1; the decision 2): the 'not testable' path (train_guard.py exit 3) reproduced
as the review did it, summarize() on runs/DRYRUN_e4rev2 with milp={} and the gobar.json revision 2's script wrote,
and with revision 3's file; a MILP STOP (exit 5) through the summary with a REAL failing milp_proof.py output; every
integrity check scoped to the bars it touches; the go-bar outcomes; E4.3's power figure on its row; and the command line
run the way STAGE=summary runs it (python -I -B; revision 3's first dry run failed there), with the freeze-record check.
Revision 4 (round-3 review M1, m5): a void bar is printed without its numbers (row, readings, the pass rates of an arm
that feeds only void bars, the go-bar counts); a logged integrity stop voids bars by section 9.1's table; the record's
decision on every void bar; a judge stop makes that arm's judged readings "not available"; the two-level text; "not
shown (not run)"; the smoke tasks from the freeze record. Revision 5 (round-4 review M1 (c), m4): after a stage's last
attempt a bar reads "not shown (stage failed)" exactly when one of its arms is incomplete (a complete bar is shown as
usual), train_guard's last attempt with no go-bar decision, the judge's last attempt, and in <c>_icr the bars the
base record did not void labelled "a reading (the bar stands from <c>)". Revision 6 (round-5 review minor 1): an
unfinished rule-9 re-run is used as it stands (the tasks it finished replace their first ends, the others keep them)
and "anomalies" lists the unfinished tasks.
Run: offline_env.sh .../taubench/.venv/bin/python -I -B -m pytest -q -p no:cacheprovider tests/test_summary.py"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))

import json
import shutil
import sys
from pathlib import Path

import pytest

E4 = Path(f"{_REPO_ROOT}/studies/agent_guards/e4")
sys.path.insert(0, str(E4))
sys.path.insert(1, str(E4 / "scripts"))

import summarize_e4 as S  # noqa: E402

GPT41 = Path(f"{_REPO_ROOT}/studies/agent_guards/taubench/data_untrusted/tau2-bench/extracted/"
             "tau2-bench-fc0055dc4e0a316c3f83133267fbd6faaa770992/data/tau2/results/final/"
             "gpt-4.1-2025-04-14_retail_default_gpt-4.1-2025-04-14_4trials.json")


def _scores(vals):
    return {str(i): float(v) for i, v in enumerate(vals)}


def test_paired_known_intervals():
    n = 40
    t = [str(i) for i in range(n)]
    # every task favours a: the interval collapses to the point
    r = S.paired(_scores([1] * n), _scores([0] * n), t)
    assert r["diff_points"] == 100.0 and r["ci95"] == [100.0, 100.0] and r["n_differ"] == 40
    # identical arms
    r = S.paired(_scores([1, 0] * 20), _scores([1, 0] * 20), t)
    assert r["diff_points"] == 0.0 and r["ci95"] == [0.0, 0.0] and r["n_differ"] == 0
    # 4 favourable flips of 40, none against: the review's +10.0 [2.5, 20.0] (review issue 4)
    a = [1, 1, 1, 1] + [0] * 36
    r = S.paired(_scores(a), _scores([0] * n), t)
    assert r["diff_points"] == 10.0 and r["ci95"] == [2.5, 20.0] and (r["a_better"], r["b_better"]) == (4, 0)
    # 114 tasks, net 4 favourable, none against: +3.51 [0.88, 7.02]
    t2 = [str(i) for i in range(114)]
    r = S.paired(_scores([1] * 4 + [0] * 110), _scores([0] * 114), t2)
    assert round(r["diff_points"], 2) == 3.51 and [round(x, 2) for x in r["ci95"]] == [0.88, 7.02]


def test_paired_requires_every_task():
    with pytest.raises(KeyError):
        S.paired({"0": 1.0}, {"0": 0.0, "1": 0.0}, ["0", "1"])


def _arms(a, b, extra=None):
    mk = lambda v: {"score": {t: {"db": x, "judged": x, "end": "normal"} for t, x in v.items()}}  # noqa: E731
    arms = {"A1_base": mk(a), "A0_base": mk(b)}
    if extra:
        arms.update({k: mk(v) for k, v in extra.items()})
    return arms


def test_bar_rules_and_completeness():
    t = [str(i) for i in range(40)]
    arms = _arms(_scores([1] * 4 + [0] * 36), _scores([0] * 40))
    r = S.bar_superiority(arms, "A1_base", "A0_base", t, "db", S.E43_MIN)
    assert r["computed"] and r["met"]
    r = S.bar_superiority(arms, "A1_base", "A0_base", t, "db", 12.0)       # same data, a higher bar
    assert r["computed"] and not r["met"]
    del arms["A1_base"]["score"]["7"]
    r = S.bar_superiority(arms, "A1_base", "A0_base", t, "db", S.E43_MIN)
    assert r["computed"] is False and "lacks 1 tasks" in r["why"]
    r = S.bar_superiority(arms, "A2s1_test", "A0_base", t, "db", S.E43_MIN)
    assert r["computed"] is False and "did not run" in r["why"]


def test_infra_and_timeout_ends_score_zero_and_are_excluded_in_the_reading():
    t = [str(i) for i in range(10)]
    arms = _arms(_scores([1] * 10), _scores([1] * 10))
    arms["A1_base"]["score"]["3"].update({"db": 0.0, "end": "infra"})
    arms["A0_base"]["score"]["5"].update({"db": 0.0, "end": "timeout"})
    r = S.paired({k: v["db"] for k, v in arms["A1_base"]["score"].items()},
                 {k: v["db"] for k, v in arms["A0_base"]["score"].items()}, t)
    assert r["a_better"] == 1 and r["b_better"] == 1            # both ends score 0 in the registered bar
    keep = S.excluding_bad_ends(arms, ["A1_base", "A0_base"], t)
    assert keep == [x for x in t if x not in ("3", "5")]


def test_e45_not_informative_and_margin():
    t = [str(i) for i in range(40)]
    same = _scores([1, 0] * 20)
    arms = _arms(same, same, {"A2s1_test": same, "A3_test": same})
    r = S.bar_noninferior(arms, "A2s1_test", "A3_test", t, "db")
    assert r["non_inferior"] is None and r["verdict"].startswith("not informative")
    a2 = _scores([1] * 4 + [0] * 36)
    arms = _arms(same, same, {"A2s1_test": a2, "A3_test": _scores([0] * 40)})
    r = S.bar_noninferior(arms, "A2s1_test", "A3_test", t, "db")
    assert r["non_inferior"] is True and r["ci95"][0] > S.E45_MARGIN
    a3 = _scores([0] * 4 + [1] * 4 + [0] * 32)                  # A3 better on 4 tasks, A2 on none
    arms = _arms(same, same, {"A2s1_test": _scores([0] * 40), "A3_test": a3})
    r = S.bar_noninferior(arms, "A2s1_test", "A3_test", t, "db")
    assert r["non_inferior"] is False and r["verdict"] == "not shown non-inferior"


def test_two_level_interval():
    t = [str(i) for i in range(40)]
    ones, zeros = _scores([1] * 40), _scores([0] * 40)
    r = S.two_level([ones, ones, ones], [zeros], t)
    assert r["ci95"] == [100.0, 100.0] and r["diff_points_all_seeds"] == 100.0 and r["n_seeds"] == [3, 1]
    # seeds that disagree widen the interval beyond the single-seed one
    s1 = _scores([1] * 6 + [0] * 34)
    s2 = _scores([0] * 40)
    s3 = _scores([1] * 12 + [0] * 28)
    one = S.paired(s1, zeros, t)
    two = S.two_level([s1, s2, s3], [zeros], t)
    assert two["ci95"][1] - two["ci95"][0] > one["ci95"][1] - one["ci95"][0]
    assert two["diff_points_all_seeds"] == pytest.approx(100 * (6 + 0 + 12) / 3 / 40)


# ------------------------------------------------------------------------------------- end to end on stand-in data
def _guard_log_from_trajectory(sim, arm_dir: Path, arm: str, inject: str | None = None):
    """Stand-in guard log: one allow verdict per executed call, with the facts recomputed from the trajectory.
    inject="code" adds one denied read verdict (code == veto, E4-R1), inject="learned" one learned-only denial, so that
    IC5 (A1) or IC6 (the test arms) compare no task (revision 4: a void bar's numbers are withheld, so the numeric
    path needs a layout in which every check holds)."""
    from e4guard import bits as B
    from e4guard import models, rules
    from e4guard.facts import events_from_message_dicts, replay_facts

    with open(arm_dir / "guard_log.jsonl", "a") as fh:
        fh.write(json.dumps({"event": "agent_init", "task": sim["task_id"], "arm": arm}) + "\n")
        for _, e, f, _ in replay_facts(events_from_message_dicts(sim["messages"])):
            x = B.encode(f)
            fh.write(json.dumps({"event": "verdict", "task": sim["task_id"], "arm": arm,
                                 "call": {"id": e["id"], "name": e["name"], "args": e.get("args")}, "allow": True,
                                 "code_rules": rules.rule_ids(f), "veto_rules": B.veto_rules(x),
                                 "veto_clauses": B.veto_fires(x), "learned": None,
                                 "monitor": {"strict_r3a": rules.strict_r3a_fires(f)}, "is_write": f["is_write"],
                                 "x": models.bits_string(f), "facts_sha256": models.facts_sha256(f)}) + "\n")
        if inject:
            r = ["E4-R1"] if inject == "code" else []
            fh.write(json.dumps({"event": "verdict", "task": sim["task_id"], "arm": arm,
                                 "call": {"id": "injected", "name": "get_order_details", "args": {}}, "allow": False,
                                 "code_rules": r, "veto_rules": r, "veto_clauses": [], "learned": None,
                                 "monitor": {}, "is_write": False, "x": "", "facts_sha256": None}) + "\n")


def _standin_campaign(root: Path, layout: dict, tasks: set | None = None, inject: dict | None = None):
    d = json.loads(GPT41.read_text())
    for arm, trial in layout.items():
        ad = root / arm
        ad.mkdir(parents=True)
        sims = [s for s in d["simulations"] if s.get("trial") == trial and (tasks is None or str(s["task_id"]) in tasks)]
        (ad / "results.json").write_text(json.dumps({"tasks": d["tasks"], "simulations": sims}))
        for s in sims:
            _guard_log_from_trajectory(s, ad, arm, (inject or {}).get(arm))


LAYOUT = {"A0_base": 0, "A1_base": 1, "A2s1_test": 2, "A2s2_test": 3, "A2s3_test": 2, "A3_test": 3, "A3t_test": 0}


@pytest.mark.skipif(not GPT41.exists(), reason="stand-in file missing")
def test_summarize_end_to_end_on_standin_data():
    """The stand-in's trajectories (gpt-4.1, old task versions; code test only). Layout 1: only tasks with no executed
    violation in any arm's trial (so IC1 holds), injected denials so that IC5 and IC6 compare nothing: every check holds
    and every bar is computed with its numbers from real trajectories. Layout 2: A1 without the injected denials, so
    IC5 compares A1 (trial 1) with A0 (trial 0), which differ: every bar is VOID and printed without numbers."""
    root = E4 / ".tmp" / "test_summary_standin" / "STANDIN_summary"
    shutil.rmtree(root.parent, ignore_errors=True)
    d = json.loads(GPT41.read_text())
    sp = json.loads(Path(S.SPLIT_FILE).read_text())
    clean = set()
    for t in [str(x) for x in sp["base"]]:
        sims = [s for s in d["simulations"] if str(s["task_id"]) == t and s.get("trial") in set(LAYOUT.values())]
        if len(sims) == 4 and not any(S.executed_violations(s) for s in sims):
            clean.add(t)
    base = [str(t) for t in sp["base"] if str(t) in clean][:24]
    test = [str(t) for t in sp["test"] if str(t) in clean][:12]
    assert len(base) == 24 and len(test) >= 8, (len(base), len(test))
    keep = set(base) | set(test)
    inject = {"A1_base": "code", **{a: "learned" for a in LAYOUT if a.endswith("_test")}}
    _standin_campaign(root, LAYOUT, keep, inject)
    gob = {"GO": True, "outcome": "go", "decision": "GO (stand-in)", "models": {"tm_seed1": {"G": 9, "L": 1}},
           "power": {"at_observed_rates": {"tm_seed1": 0.25}, "at_go_bar_corner_G8_L4": 0.12}}
    eq = {"mismatches": 0}
    milp = {f"milp_tm_seed{k}.json": {"summary": {"guarded": {"proved": 13}}, "pre_D_domain_holds": [50, 50]}
            for k in (1, 2, 3)}
    out = S.summarize(root, sp, S.SMOKE_TASKS, gob, eq, milp, base_tasks=base, test_tasks=test)
    b, ic = out["bars"], out["integrity"]
    assert ic["checks"] == {"IC1": True, "IC2a": True, "IC2b": True, "IC3_IC4": True, "IC5": True, "IC6": True}, ic["checks"]
    assert ic["IC5_A1_equals_A0_where_A1_denied_nothing"]["tasks_checked"] == 0 and ic["all_ok"] is True
    assert b["E4.2"]["computed"] and b["E4.2"]["n_tasks"] == 24 and "flag" not in b["E4.2"]
    assert b["E4.3"]["computed"] and b["E4.3"]["n_tasks"] == len(test)
    assert b["E4.5"]["computed"] and "verdict" in b["E4.5"]
    assert b["E4.2"]["n_differ"] > 0                                 # other trials: real differences, a real interval
    assert out["readings"]["E4.3_two_level"]["n_seeds"] == [3, 1]
    assert out["readings"]["E4.2_db_without_smoke_tasks"]["n_tasks"] == len([t for t in base if t not in S.SMOKE_TASKS])
    rows = {r["bar"]: r for r in out["table"]}
    assert all("diff_points" in rows[k] and "ci95" in rows[k] for k in S.BARS)
    assert rows["E4.3"]["power"]["low_power"] is True and rows["E4.3"]["power"]["at_observed_go_bar_rates_tm_seed1"] == 0.25
    assert out["arms"]["A1_base"]["pass1_db"] is not None
    # layout 2: A1's logs without the injected denials: IC5 compares trial 1 with trial 0 and fails: every bar void
    shutil.rmtree(root.parent, ignore_errors=True)
    _standin_campaign(root, LAYOUT, keep, {a: "learned" for a in LAYOUT if a.endswith("_test")})
    out2 = S.summarize(root, sp, S.SMOKE_TASKS, gob, eq, milp, base_tasks=base, test_tasks=test)
    b2, i2 = out2["bars"], out2["integrity"]
    assert i2["checks"]["IC5"] is False and i2["IC5_A1_equals_A0_where_A1_denied_nothing"]["trajectory_differs"]
    assert all(b2[k]["void"] is True and b2[k]["numbers_withheld"] is True and "IC5" in b2[k]["flag"] for k in S.BARS)
    rows2 = {r["bar"]: r for r in out2["table"]}
    assert all(rows2[k]["outcome"] == "VOID (integrity failure)" and "diff_points" not in rows2[k]
               and "ci95" not in rows2[k] for k in S.BARS)
    assert all(not v.get("computed") and "withheld" in v["why"] for k, v in out2["readings"].items()
               if k.startswith(("E4.2_", "E4.3_", "E4.5_")))
    assert all(out2["arms"][nm]["pass1_db"] is None for nm in out2["arms"])          # every arm feeds only void bars
    assert rows2["E4.3"]["power"]["at_observed_go_bar_rates_tm_seed1"] is None
    assert out2["gobar"]["G"] is None and "withheld" in out2["gobar"]
    assert "E4.2_" not in json.dumps({k: v for k, v in b2.items()})                   # no reading leaks into a bar
    shutil.rmtree(root.parent, ignore_errors=True)


# ----------------------------------------------- revision 3: the go-bar's outcomes and each check's scope
DRY2 = E4 / "runs" / "DRYRUN_e4rev2"                  # revision 2's end-to-end dry run (read only)
DRY_BASE, DRY_TEST, DRY_SMOKE = ["0", "1", "2", "3", "4", "5", "9", "12"], ["5", "9", "12"], ["0", "1", "2"]


def test_gate_outcome_reads_every_form():
    assert S.gate_outcome(None) == "not run"
    assert S.gate_outcome({"GO": False, "decision": "NOT TESTABLE: too few veto-clean rows of a class"}) == "not testable"
    assert S.gate_outcome({"GO": False, "outcome": "not testable"}) == "not testable"
    assert S.gate_outcome({"GO": False, "outcome": "void"}) == "void"
    assert S.gate_outcome({"GO": False, "decision": "NO-GO: ..."}) == "no-go"
    assert S.gate_outcome({"GO": True, "outcome": "go"}) == "go"
    assert S.gate_outcome({"GO": False, "outcome": "go"}) == "no-go"       # an inconsistent file never counts as GO
    assert S.gate_outcome({"GO": True, "outcome": "no-go"}) == "no-go"
    assert S.gate_outcome({"GO": True, "decision": "GO (revision 2 file)"}) == "go"


def _summ_dry(gobar, milp, eq=None):
    sp = json.loads(Path(S.SPLIT_FILE).read_text())
    return S.summarize(DRY2, sp, DRY_SMOKE, gobar, {"mismatches": 0} if eq is None else eq, milp,
                       base_tasks=DRY_BASE, test_tasks=DRY_TEST)


@pytest.mark.skipif(not (DRY2 / "A1_base" / "results.json").exists(), reason="revision 2's dry run folder missing")
def test_not_testable_path_reviewers_case_on_the_rev2_dry_run():
    """The review's reproduction: summarize() on runs/DRYRUN_e4rev2 with milp={} and the gobar.json that revision 2's
    run_e4_gpu.sh wrote on train_guard.py exit 3. Revision 2 flagged EVERY bar (E4.2 included) INTEGRITY FAILURE and
    read E4.3/E4.5 'go-bar NO-GO (or not run)'. Now: E4.2 computed with no flag; E4.3 and E4.5 'not testable';
    IC2(b) and IC6 not applicable; the test-arm folders present there are not read (an anomaly is noted)."""
    for gob in ({"GO": False, "decision": "NOT TESTABLE: too few veto-clean rows of a class"},          # revision 2's
                {"GO": False, "outcome": "not testable", "train_guard_exit": 3, "power": None,           # revision 3's
                 "decision": "NOT TESTABLE: fewer than 10 veto-clean, deduplicated train rows of a class"}):
        out = _summ_dry(gob, {})
        b, ic = out["bars"], out["integrity"]
        assert out["go_bar_outcome"] == "not testable"
        assert b["E4.2"]["computed"] is True and "flag" not in b["E4.2"] and not b["E4.2"].get("void")
        for k in ("E4.3", "E4.5"):
            assert b[k]["computed"] is False and b[k]["why"].startswith("not testable"), b[k]
            assert "flag" not in b[k]
        assert ic["checks"]["IC2b"] is None and ic["checks"]["IC6"] is None and ic["IC2b"]["why"].startswith(
            "not applicable")
        assert ic["all_ok"] is True and ic["by_bar"]["E4.2"]["failed"] == []
        assert set(out["arms"]) == {"A0_base", "A1_base"} and out["anomalies"]
        rows = {r["bar"]: r for r in out["table"]}
        assert rows["E4.3"]["outcome"].startswith("not testable") and "flag" not in rows["E4.2"]
        assert rows["E4.3"]["power"]["low_power"] is True and rows["E4.3"]["power"]["at_observed_go_bar_rates_tm_seed1"] is None


@pytest.mark.skipif(not (DRY2 / "A1_base" / "results.json").exists(), reason="revision 2's dry run folder missing")
def test_milp_stop_exit5_voids_only_e43_e45(tmp_path):
    """A MILP STOP (STAGE=train_guard exit 5) through the summary, with a REAL failing milp_proof.py output: a rows file
    holding one vector outside the domain makes pre-check D fail (milp_proof exits 1). gobar.json is the one
    revision 3's script writes on exit 5. E4.2 stands with no flag; E4.3 and E4.5 are void (IC2(b))."""
    import subprocess

    import numpy as np

    from e4guard import bits as B

    x = np.zeros(B.N_BITS, np.uint8)                   # a one-hot group broken on purpose: two tools at once
    tools = [i for i, n in enumerate(B.NAMES) if n.startswith("tool=")][:2]
    x[tools] = 1
    assert not B.rows_hold(B.domain_rows(), x[None])[0]
    rows = tmp_path / "bad_rows.json"
    rows.write_text(json.dumps({"names": B.NAMES, "rows": [{"x": x.tolist(), "label": 0}]}))
    out_p = tmp_path / "milp_tm_seed1.json"
    dlpy = f"{_REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python"
    p = subprocess.run([dlpy, "-I", "-B", str(E4 / "scripts/milp_proof.py"), "--model",
                        str(E4 / "models/standin_rev2/tm_seed1.json"), "--rows", str(rows), "--out", str(out_p),
                        "--n-votes", "1", "--n-winner", "1", "--threads", "2"],
                       capture_output=True, text=True, timeout=600,
                       env={"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "PATH": "/usr/bin:/bin"})
    assert p.returncode == 1, p.stderr[-600:]          # milp_proof's own failure exit (the stage turns it into 5)
    milp = {"milp_tm_seed1.json": json.loads(out_p.read_text())}
    gob = {"GO": False, "outcome": "void", "failed_check": "IC2b", "seed": 1, "power": None,
           "decision": "STOP: IC2(b) failed (the MILP check, TM seed 1: ...); E4.3 and E4.5 are void (section 9.1)"}
    out = _summ_dry(gob, milp)
    b, ic = out["bars"], out["integrity"]
    assert ic["checks"]["IC2b"] is False and ic["IC2b"]["per_seed"]["1"]["ok"] is False
    assert ic["IC2b"]["per_seed"]["2"]["why"] == "missing"
    assert b["E4.2"]["computed"] is True and "flag" not in b["E4.2"]
    for k in ("E4.3", "E4.5"):
        assert b[k]["computed"] is False and b[k]["void"] is True and "why" not in b[k]   # revision 4: no text, no number
        assert "IC2b" in b[k]["flag"]
    assert ic["by_bar"]["E4.2"]["failed"] == [] and ic["by_bar"]["E4.3"]["failed"] == ["IC2b"]
    rows_t = {r["bar"]: r for r in out["table"]}
    assert rows_t["E4.3"]["outcome"].startswith("VOID (integrity failure)")


@pytest.mark.skipif(not (DRY2 / "A1_base" / "results.json").exists(), reason="revision 2's dry run folder missing")
def test_scope_ic2a_touches_every_bar_and_no_go_carries_power():
    gob = {"GO": False, "outcome": "no-go", "decision": "NO-GO", "models": {"tm_seed1": {"G": 3, "L": 5}},
           "power": {"at_observed_rates": {"tm_seed1": 0.006}, "at_go_bar_corner_G8_L4": 0.12,
                     "effect_for_80pct_power": [{"net_points": 13.36}]}}
    milp = {f"milp_tm_seed{k}.json": {"summary": {"guarded": {"proved": 13}}, "pre_D_domain_holds": [9, 9]}
            for k in (1, 2, 3)}
    out = _summ_dry(gob, milp)
    assert out["bars"]["E4.3"]["why"].startswith("NO-GO at the dev go-bar") and "0.6%" in out["bars"]["E4.3"]["why"]
    assert out["integrity"]["checks"]["IC2b"] is True and "flag" not in out["bars"]["E4.2"]
    row = {r["bar"]: r for r in out["table"]}["E4.3"]
    assert row["power"]["net_points_for_80pct_power_nothing_against"] == 13.36
    out = _summ_dry(gob, milp, eq={"mismatches": 1})            # IC2(a) fails: every bar is void
    assert out["integrity"]["checks"]["IC2a"] is False
    assert all(out["integrity"]["by_bar"][k]["failed"] == ["IC2a"] for k in S.BARS)
    assert "flag" in out["bars"]["E4.2"] and out["bars"]["E4.2"]["void"] is True


def test_integrity_rerun_campaign_reads_not_shown(tmp_path):
    """In the one integrity re-run (<campaign>_icr) a failed check reads 'not shown (integrity failure)'."""
    root = tmp_path / "e4reg_00000000_icr"
    root.mkdir()
    sp = json.loads(Path(S.SPLIT_FILE).read_text())
    out = S.summarize(root, sp, S.SMOKE_TASKS, {"GO": False, "outcome": "not testable"}, {"mismatches": 1}, {},
                      base_tasks=["0"], test_tasks=["5"])
    assert out["bars"]["E4.2"]["computed"] is False and out["bars"]["E4.2"]["void"] is True
    assert "not shown (integrity failure)" in out["bars"]["E4.2"]["flag"]
    assert {r["bar"]: r for r in out["table"]}["E4.2"]["outcome"].startswith("not shown (integrity failure)")


PY = f"{_REPO_ROOT}/studies/agent_guards/taubench/.venv/bin/python"


def _summary_cli(tmp_path, *extra):
    """summarize_e4.py the way STAGE=summary runs it: python -I -B (the script's folder is NOT on sys.path)."""
    import subprocess

    env = {"PATH": "/usr/bin:/bin", "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
           "OPENBLAS_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false", "TMPDIR": str(E4 / ".tmp")}
    cmd = [PY, "-I", "-B", str(E4 / "scripts/summarize_e4.py"), "--campaign", DRY2.name,
           "--base-tasks", *DRY_BASE, "--test-tasks", *DRY_TEST, "--smoke-tasks", *DRY_SMOKE, *extra]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env, cwd=str(tmp_path))


@pytest.mark.skipif(not (DRY2 / "A1_base" / "results.json").exists(), reason="revision 2's dry run folder missing")
def test_summary_command_line_runs_isolated_as_the_stage_runs_it(tmp_path):
    """Revision 3's first end-to-end dry run stopped at STAGE=summary: under python -I, summarize_e4.py could not import
    check_pins (ModuleNotFoundError, exit 1). Now, on revision 2's dry-run folder (read only): exit 7 with no freeze
    record; exit 0 and a full summary with a dry record that names the registration, its sha256 and the campaign; exit
    7 again once the registration differs from the record (one byte appended)."""
    import hashlib

    reg = tmp_path / "registration_dry.md"
    reg.write_text("a dry registration text for the summary's command-line test\n")
    out = tmp_path / "summary.json"
    p = _summary_cli(tmp_path, "--registration", str(reg), "--frozen", str(tmp_path / "no_record.json"),
                     "--out", str(out))
    assert "ModuleNotFoundError" not in p.stderr, p.stderr[-800:]
    assert p.returncode == 7 and "not frozen" in p.stdout and not out.exists(), (p.returncode, p.stderr[-800:])
    rec = tmp_path / "E4_frozen_dry.json"
    rec.write_text(json.dumps({"registration": str(reg.resolve()),
                               "registration_sha256": hashlib.sha256(reg.read_bytes()).hexdigest(),
                               "campaign": DRY2.name}))
    p = _summary_cli(tmp_path, "--registration", str(reg), "--frozen", str(rec), "--out", str(out))
    assert p.returncode == 0, (p.returncode, p.stdout[-800:], p.stderr[-1500:])
    sm = json.loads(out.read_text())
    assert [r["bar"] for r in sm["table"]] == ["E4.2", "E4.3", "E4.5"]
    assert sm["bars"]["E4.2"]["computed"] is True and set(sm["integrity"]["checks"]) == set(S.SCOPE)
    assert sm["inputs"]["frozen_record"] == str(rec) and "power" in {r["bar"]: r for r in sm["table"]}["E4.3"]
    with open(reg, "a") as f:
        f.write("x")
    out2 = tmp_path / "summary2.json"
    p = _summary_cli(tmp_path, "--registration", str(reg), "--frozen", str(rec), "--out", str(out2))
    assert p.returncode == 7 and "differs from the frozen" in p.stdout and not out2.exists(), p.stdout[-800:]


# ----------------------------------------------- revision 4: void bars without numbers; integrity stops; the record
GO = {"GO": True, "outcome": "go", "decision": "GO (dry)", "models": {"tm_seed1": {"G": 9, "L": 1}},
      "power": {"at_observed_rates": {"tm_seed1": 0.2}, "at_go_bar_corner_G8_L4": 0.12}}
MILP_OK = {f"milp_tm_seed{k}.json": {"summary": {"guarded": {"proved": 13}}, "pre_D_domain_holds": [9, 9]}
           for k in (1, 2, 3)}
STOP = "2026-10-22T10:00:00Z [a2a3] STOP [identity_changed] run=A2s1_test: e4_run.py for c/A2s1_test failed (exit 2): x"


def _summ_go(log="", record=None, eq=None, root=None):
    sp = json.loads(Path(S.SPLIT_FILE).read_text())
    return S.summarize(root or DRY2, sp, DRY_SMOKE, GO, {"mismatches": 0} if eq is None else eq, MILP_OK,
                       base_tasks=DRY_BASE, test_tasks=DRY_TEST, log_text=log, record=record)


@pytest.mark.skipif(not (DRY2 / "A1_base" / "results.json").exists(), reason="revision 2's dry run folder missing")
def test_void_bars_carry_no_numbers():
    out = _summ_go()
    assert out["integrity"]["all_ok"] is True and out["integrity"]["void_bars"] == []
    rows = {r["bar"]: r for r in out["table"]}
    assert all("diff_points" in rows[k] for k in S.BARS) and out["arms"]["A2s1_test"]["pass1_db"] is not None
    out = _summ_go(eq={"mismatches": 1})                         # IC2(a) fails: every bar void
    rows = {r["bar"]: r for r in out["table"]}
    for k in S.BARS:
        assert rows[k]["outcome"] == "VOID (integrity failure)" and not {"diff_points", "ci95", "n_tasks"} & set(rows[k])
        assert out["bars"][k]["numbers_withheld"] and not {"diff_points", "ci95", "a", "b"} & set(out["bars"][k])
        assert "integrity_failure.py" in out["bars"][k]["flag"]   # no record yet: the record decides
    assert all("withheld" in v.get("why", "") for k, v in out["readings"].items())
    assert all(a["pass1_db"] is None for a in out["arms"].values()) and out["gobar"]["G"] is None


@pytest.mark.skipif(not (DRY2 / "A1_base" / "results.json").exists(), reason="revision 2's dry run folder missing")
def test_a_logged_integrity_stop_and_the_records_decision():
    out = _summ_go(log="2026-10-22T09:00:00Z [a2a3] pins ok (x)\n" + STOP + "\n")
    b, ic = out["bars"], out["integrity"]
    assert ic["checks"]["IC6"] is True and ic["stop_void_bars"] == ["E4.3", "E4.5"] and ic["all_ok"] is False
    assert [e["line"] for e in ic["stop_events"]] == [STOP]
    assert "flag" not in b["E4.2"] and b["E4.2"]["computed"] is True and "diff_points" in b["E4.2"]
    for k in ("E4.3", "E4.5"):
        assert b[k]["void"] is True and "stop identity_changed at a2a3" in b[k]["flag"]
    assert out["arms"]["A2s1_test"]["pass1_db"] is None and out["arms"]["A1_base"]["pass1_db"] is not None
    assert out["readings"]["E4.2_db_without_smoke_tasks"]["computed"] is True
    assert "withheld" in out["readings"]["E4.3_two_level"]["why"]
    rec = {"failed_checks": ["identity_changed"], "void_bars": ["E4.3", "E4.5"], "rerun_required": True,
           "decision": "the one integrity re-run DRYRUN_e4rev2_icr IS run ...", "logged_utc": "2026-10-22T12:00:00Z"}
    out = _summ_go(log=STOP + "\n", record=rec)
    rows = {r["bar"]: r for r in out["table"]}
    assert rows["E4.3"]["outcome"] == "VOID (integrity failure)"
    assert "replaced by the integrity re-run" in rows["E4.3"]["flag"]
    assert not any("void bars" in a for a in out["anomalies"])
    out = _summ_go(log=STOP + "\n", record={**rec, "rerun_required": False, "decision": "the logged fix changes ..."})
    rows = {r["bar"]: r for r in out["table"]}
    assert rows["E4.3"]["outcome"] == "not shown (integrity failure)" and "diff_points" not in rows["E4.3"]
    out = _summ_go(log=STOP + "\n", record={**rec, "void_bars": ["E4.2", "E4.3", "E4.5"]})
    assert any("void bars" in a for a in out["anomalies"])        # a record that does not match the derived voids


@pytest.mark.skipif(not (DRY2 / "A1_base" / "results.json").exists(), reason="revision 2's dry run folder missing")
def test_a_judge_stop_voids_no_bar_but_the_judged_readings():
    log = "2026-10-22T10:00:00Z [judge] STOP [identity_changed] run=judge:A1_base: the judge failed (exit 2): x\n"
    out = _summ_go(log=log)
    assert out["integrity"]["void_bars"] == [] and out["integrity"]["judged_readings_not_available"] == ["A1_base"]
    assert out["bars"]["E4.2"]["computed"] is True and "flag" not in out["bars"]["E4.2"]
    assert out["arms"]["A1_base"]["pass1_judged_self_judged_unvalidated"] is None
    assert "not available" in out["arms"]["A1_base"]["judged_readings"]
    assert out["readings"]["E4.2_judged_self_judged_unvalidated"]["computed"] is False


@pytest.mark.skipif(not (DRY2 / "A1_base" / "results.json").exists(), reason="revision 2's dry run folder missing")
def test_two_level_text_when_seeds_are_cut(tmp_path):
    """Round-3 review m5(b): the code's text and section 10.6's text are the same."""
    root = tmp_path / "DRYRUN_seedcut"
    shutil.copytree(DRY2, root, ignore=shutil.ignore_patterns("A2s2_test*", "A2s3_test*", "replay.sqlite"))
    out = _summ_go(root=root)
    w = out["readings"]["E4.3_two_level"]["why"]
    assert w == "not available (seeds cut or incomplete): seeds available ['A2s1_test'] (need 2 or more)"
    reg = (E4 / "REGISTRATION_E4.md").read_text()
    assert '"not available (seeds cut or incomplete): seeds available [...] (need 2 or more)"' in reg


def test_not_shown_not_run_and_the_smoke_tasks_default():
    row = S._table_row("E4.3", {"computed": False, "why": "arm A2s1_test did not run"})
    assert row["outcome"] == "not shown (not run): arm A2s1_test did not run"
    row = S._table_row("E4.2", {"computed": False, "why": "arm A1_base lacks 2 tasks: ['3', '9']"})
    assert row["outcome"].startswith("arm A1_base lacks")               # incomplete (rule 9): not "not run"
    src = (E4 / "scripts/summarize_e4.py").read_text()                  # the command line's default smoke tasks
    assert 'fz.get("smoke_tasks")' in src and '"--smoke-tasks", nargs="*", default=None' in src


# ------------------------------------------- revision 5: "not shown (stage failed)"; the _icr reading label (m4)
def _fails(stage, run, k):
    return "".join(f"2026-10-22T11:0{i}:00Z [{stage}] STOP [client] run={run}: e4_run.py for c/{run} failed (exit 1): x\n"
                   for i in range(k))


def _drop(root, arm, task):
    p = root / arm / "results.json"
    d = json.loads(p.read_text())
    d["simulations"] = [x for x in d["simulations"] if str(x["task_id"]) != task]
    p.write_text(json.dumps(d))


def _rows(out):
    return {r["bar"]: r for r in out["table"]}


@pytest.mark.skipif(not (DRY2 / "A1_base" / "results.json").exists(), reason="revision 2's dry run folder missing")
def test_after_the_last_attempt_only_a_bar_with_an_incomplete_arm_is_not_shown(tmp_path):
    root = tmp_path / "DRYRUN_stagefail"
    shutil.copytree(DRY2, root, ignore=shutil.ignore_patterns("replay.sqlite"))
    _drop(root, "A3_test", "9")                                   # a2a3 stopped inside A3 (A2s1 had finished)
    r = _rows(_summ_go(root=root, log=_fails("a2a3", "A3_test", 2)))             # 2 failed attempts: a third runs
    assert r["E4.3"]["computed"] and "diff_points" in r["E4.3"]
    assert r["E4.5"]["outcome"].startswith("arm A3_test lacks 1 tasks")
    out = _summ_go(root=root, log=_fails("a2a3", "A3_test", 3))                  # the last attempt failed
    r = _rows(out)
    assert r["E4.3"]["computed"] and "diff_points" in r["E4.3"] and "flag" not in r["E4.3"]   # complete: shown
    assert r["E4.5"]["outcome"].startswith("not shown (stage failed): a2a3 failed its 3 attempts (section 11.6); "
                                           "arm A3_test lacks 1 tasks") and "diff_points" not in r["E4.5"]
    assert r["E4.2"]["computed"] and out["stage_attempts"]["final"] == ["a2a3"]
    assert out["integrity"]["void_bars"] == []                    # a stage failure is not an integrity failure
    _drop(root, "A0_base", "1")                                   # a0's last attempt failed with task 1 missing
    r = _rows(_summ_go(root=root, log=_fails("a0", "A0_base", 3)))
    assert r["E4.2"]["outcome"].startswith("not shown (stage failed): a0 failed its 3 attempts")
    assert r["E4.3"]["computed"] and r["E4.5"]["outcome"].startswith("arm A3_test lacks")   # a2a3 not final here
    sp = json.loads(Path(S.SPLIT_FILE).read_text())               # train_guard's last attempt, no go-bar decision
    tg = "".join(f"2026-10-22T11:0{i}:00Z [train_guard] STOP [train_guard]: train_guard.py failed (exit 1)\n"
                 for i in range(3))
    r = _rows(S.summarize(DRY2, sp, DRY_SMOKE, None, {"mismatches": 0}, {}, base_tasks=DRY_BASE, test_tasks=DRY_TEST,
                          log_text=tg))
    assert all(r[k]["outcome"].startswith("not shown (stage failed): train_guard failed its 3 attempts")
               for k in ("E4.3", "E4.5")) and r["E4.2"]["computed"]
    r = _rows(S.summarize(DRY2, sp, DRY_SMOKE, None, {"mismatches": 0}, {}, base_tasks=DRY_BASE, test_tasks=DRY_TEST))
    assert r["E4.3"]["outcome"].startswith("not run: no go-bar decision")       # train_guard not final: not run
    root2 = tmp_path / "DRYRUN_judgefail"                         # the judge's last attempt failed before A3_test
    shutil.copytree(DRY2, root2, ignore=shutil.ignore_patterns("replay.sqlite"))
    (root2 / "A3_test" / "judged.json").unlink()
    jl = "".join(f"2026-10-22T11:0{i}:00Z [judge] STOP [client] run=judge:A3_test: the judge for A3_test failed "
                 "(exit 1): x\n" for i in range(3))
    out = _summ_go(root=root2, log=jl)
    assert "not available (the judge failed its 3 attempts" in out["arms"]["A3_test"]["judged_readings"]
    assert "judged_readings" not in out["arms"]["A2s1_test"] and _rows(out)["E4.5"]["computed"]


@pytest.mark.skipif(not (DRY2 / "A1_base" / "results.json").exists(), reason="revision 2's dry run folder missing")
def test_in_the_integrity_rerun_a_bar_the_base_record_did_not_void_is_a_reading(tmp_path):
    root = tmp_path / "DRYRUN_e4rev2_icr"
    shutil.copytree(DRY2, root, ignore=shutil.ignore_patterns("replay.sqlite"))
    sp = json.loads(Path(S.SPLIT_FILE).read_text())
    rec = {"void_bars": ["E4.3", "E4.5"], "rerun_required": True}
    r = _rows(S.summarize(root, sp, DRY_SMOKE, GO, {"mismatches": 0}, MILP_OK, base_tasks=DRY_BASE,
                          test_tasks=DRY_TEST, base_record=rec))
    assert r["E4.2"]["outcome"].startswith("a reading (the bar stands from DRYRUN_e4rev2): ")
    assert r["E4.2"]["kind"] == "reading" and "diff_points" in r["E4.2"]
    assert all("kind" not in r[k] and not r[k]["outcome"].startswith("a reading") for k in ("E4.3", "E4.5"))
    r = _rows(S.summarize(root, sp, DRY_SMOKE, GO, {"mismatches": 0}, MILP_OK, base_tasks=DRY_BASE,
                          test_tasks=DRY_TEST, base_record=None))
    assert not any(x["outcome"].startswith("a reading") for x in r.values())
    src = (E4 / "scripts/summarize_e4.py").read_text()            # main() reads the base campaign's record
    assert 'E4 / "runs" / a.campaign[: -len("_icr")] / "integrity_failure.json"' in src


# ------------------------------------------- revision 6 (round-5 review minor 1): an unfinished rule-9 re-run
@pytest.mark.skipif(not (DRY2 / "A1_base" / "results.json").exists(), reason="revision 2's dry run folder missing")
def test_an_unfinished_rerun_is_used_as_it_stands_and_listed_under_anomalies(tmp_path):
    root = tmp_path / "DRYRUN_rerunpart"
    shutil.copytree(DRY2, root, ignore=shutil.ignore_patterns("replay.sqlite"))
    a0 = json.loads((root / "A0_base" / "results.json").read_text())
    first = {str(x["task_id"]): x for x in a0["simulations"]}
    rr = root / "A0_base_rerun"
    rr.mkdir()
    (rr / "rerun_tasks.txt").write_text("1 3\n")                  # the re-run's list: tasks 1 and 3
    redo = json.loads(json.dumps(first["1"]))
    redo["termination_reason"] = "user_stop"                      # task 1 finished in the re-run
    (rr / "results.json").write_text(json.dumps({"simulations": [redo], "tasks": []}))   # task 3 did not
    out = _summ_go(root=root, log=_fails("rerun", "A0_base_rerun", 3))                    # the rerun stage is final
    an = [x for x in out["anomalies"] if "rule-9 re-run of A0_base" in x]
    assert an == ["the rule-9 re-run of A0_base is unfinished: tasks ['3'] of its list keep their first infra or "
                  "timeout ends (score 0, counted; section 3.3); the re-run stage failed its 3 attempts (final; "
                  "section 11.6)"], out["anomalies"]
    assert out["arms"]["A0_base"]["rerun_tasks"] == ["1"] and out["arms"]["A0_base"]["n"] == len(DRY_BASE)
    assert _rows(out)["E4.2"]["computed"]                         # the arm still holds every task (3.3's rule)
    out = _summ_go(root=root)                                     # not final: the re-run resumes on its list
    assert any("its stage is not final: the re-run resumes on its list" in x for x in out["anomalies"])
    (rr / "results.json").write_text(json.dumps({"simulations": [redo, first["3"]], "tasks": []}))
    assert not any("rule-9 re-run" in x for x in _summ_go(root=root)["anomalies"])          # finished: nothing listed
    (rr / "rerun_tasks.txt").unlink()
    assert S.rerun_unfinished(root / "A0_base") == []             # no list (before revision 5): nothing to compare
