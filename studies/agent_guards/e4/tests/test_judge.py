"""Tests of the post-hoc judge's scoring paths (scripts/judge_posthoc.py; review issues 3 and 11): a task with no
assertions, an early end, a judge error, an empty results list, a count mismatch, met and not met; the health-only
output; and the judge STUB end to end on stand-in data through tau2's own evaluate_nl_assertions (its generate
replaced by a stub that contacts nothing)."""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))

import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

E4 = Path(f"{_REPO_ROOT}/studies/agent_guards/e4")
sys.path.insert(0, str(E4))
sys.path.insert(1, str(E4 / "scripts"))

import judge_posthoc as J  # noqa: E402

GPT41 = Path(f"{_REPO_ROOT}/studies/agent_guards/taubench/data_untrusted/tau2-bench/extracted/"
             "tau2-bench-fc0055dc4e0a316c3f83133267fbd6faaa770992/data/tau2/results/final/"
             "gpt-4.1-2025-04-14_retail_default_gpt-4.1-2025-04-14_4trials.json")


def _sim(task="7", db=1.0, term="user_stop"):
    return NS(task_id=task, reward_info=NS(reward=db), termination_reason=term, messages=[])


def _task(n):
    return NS(evaluation_criteria=NS(nl_assertions=[f"Agent should tell the user fact {i}" for i in range(n)]))


def _never(*a):
    raise AssertionError("the judge must not be called")


def test_no_judge_task_scores_the_db_reward_without_a_call():
    r = J.judge_one(_task(0), _sim(db=1.0), False, _never)
    assert r["nl"] == 1.0 and r["judged"] == 1.0 and "judge_error" not in r and "parsed" not in r
    r = J.judge_one(_task(0), _sim(db=0.0), False, _never)
    assert r["judged"] == 0.0


def test_early_end_is_not_judged_and_scores_zero():
    r = J.judge_one(_task(1), _sim(db=0.0, term="max_steps"), True, _never)
    assert r["judged"] == 0.0 and r["nl"] is None and "ended early" in r["note"]
    r = J.judge_one(_task(1), _sim(db=0.0, term="infrastructure_error"), True, _never)
    assert r["judged"] == 0.0


def test_judge_error_empty_results_and_count_mismatch():
    def raises(traj, a):
        raise json.JSONDecodeError("bad", "x", 0)

    r = J.judge_one(_task(1), _sim(), True, raises)
    assert r["judged"] == 0.0 and r["nl"] == 0.0 and "judge_error" in r and r["parsed"] is False
    r = J.judge_one(_task(2), _sim(), True, lambda traj, a: [])           # empty 'results': tau2 would give 1.0
    assert r["judged"] == 0.0 and "count 0 != assertions count 2" in r["judge_error"] and r["count_match"] is False
    r = J.judge_one(_task(2), _sim(), True, lambda traj, a: [NS(met=True)])
    assert r["judged"] == 0.0 and "judge_error" in r


def test_met_and_not_met():
    r = J.judge_one(_task(2), _sim(db=1.0), True, lambda traj, a: [NS(met=True), NS(met=True)])
    assert r["judged"] == 1.0 and r["nl"] == 1.0 and r["count_match"]
    r = J.judge_one(_task(2), _sim(db=1.0), True, lambda traj, a: [NS(met=True), NS(met=False)])
    assert r["judged"] == 0.0 and r["nl"] == 0.0 and "judge_error" not in r
    r = J.judge_one(_task(1), _sim(db=0.0), True, lambda traj, a: [NS(met=True)])
    assert r["judged"] == 0.0                                           # DB x NL


def test_health_only_output_has_no_score():
    rows = [J.judge_one(_task(1), _sim(), True, lambda t, a: [NS(met=True)]),
            J.judge_one(_task(2), _sim(), True, lambda t, a: [])]
    s = J.summarize(rows, health_only=True)
    assert set(s) == {"health"} and s["health"]["judge_calls"] == 2 and s["health"]["count_match"] == 1
    assert s["health"]["parse_rate"] == 1.0 and s["health"]["count_match_share"] == 0.5
    full = J.summarize(rows, health_only=False)
    assert full["judge_errors"] == 1 and full["pass1_judged"] == 0.5


@pytest.mark.skipif(not GPT41.exists(), reason="stand-in file missing")
def test_judge_stub_end_to_end_on_standin_data():
    """tau2's own evaluate_nl_assertions, its `generate` stubbed (nothing is contacted): a reply with every result
    met, a reply with an EMPTY results list (judge error under our rule), and a reply that is not JSON."""
    from tau2.data_model.message import AssistantMessage
    from tau2.data_model.simulation import Results
    from tau2.evaluator import evaluator_nl_assertions as nl
    from tau2.registry import registry

    res = Results.load(GPT41)
    need = set(json.loads((E4 / "results/judge_tasks.json").read_text())["summary"]["judge_tasks_static"]["ids"])
    sims = [s for s in res.simulations if s.trial == 0][:40]
    # the v1.0.1 task versions, as judge_posthoc.py loads them (the file's embedded old versions have no assertions)
    tasks = {str(t.id): t for t in registry.get_tasks_loader("retail")(None)}
    old = {str(t.id): t for t in res.tasks}
    r_old = J.judge_one(old[next(iter(sorted(need, key=int)))], sims[0], True, lambda t, a: [])
    assert "judge_error" in r_old or r_old["note"].startswith("ended early")
    mode = {"m": "all"}

    def fake_generate(model, messages, **kw):
        up = messages[-1].content
        import ast

        items = ast.literal_eval(up.split("expectedOutcomes:", 1)[1].strip())
        if mode["m"] == "all":
            body = {"results": [{"expectedOutcome": x, "reasoning": "stub", "metExpectation": True} for x in items]}
        elif mode["m"] == "empty":
            body = {"results": []}
        else:
            return AssistantMessage(role="assistant", content="not json")
        return AssistantMessage(role="assistant", content=json.dumps(body))

    saved = nl.generate
    try:
        nl.generate = fake_generate
        ev = nl.NLAssertionsEvaluator.evaluate_nl_assertions
        out = {}
        for m in ("all", "empty", "bad"):
            mode["m"] = m
            out[m] = J.judge_rows(sims, tasks, need, ev)
    finally:
        nl.generate = saved
    judged_tasks = [r for r in out["all"] if "parsed" in r]
    assert judged_tasks, "some stand-in sims need the judge"
    assert all(r["count_match"] and r["judged"] == r["db"] for r in judged_tasks)
    # tasks that need no judge score their DB reward in every mode; tasks that do are errors in 'empty' and 'bad'
    for m in ("empty", "bad"):
        for r in out[m]:
            if "parsed" in r:
                assert r["judged"] == 0.0 and "judge_error" in r
            else:
                assert r["judged"] == (0.0 if r.get("note", "").startswith("ended early") else r["db"])
    h = J.summarize(out["empty"], health_only=True)["health"]
    assert h["count_match"] == 0 and h["judge_errors"] == len(judged_tasks)
