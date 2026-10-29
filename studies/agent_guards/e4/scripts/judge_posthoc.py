"""Post-hoc NL-assertion judge for E4 (GPU via the shared llama-server; NOT RUN by the preparer against a real server).

tau2's own NLAssertionsEvaluator.evaluate_nl_assertions, with its module globals pointed at the local Qwen (the same
model that acted: the judged score is SELF-JUDGED and UNVALIDATED and is reported only as such, next to the judge-free
DB score, never as a bar). Only the tasks that need the judge (results/judge_tasks.json: 40 retail tasks, 11 in the
test split) are judged. Scoring paths (judge_one; unit-tested in tests/test_judge.py):
  - a task with no NL assertions:   nl 1.0 with no LLM call, judged = the DB reward;
  - a simulation that ended early:  not sent to the judge, judged 0 (its DB reward is 0 already);
  - a judge reply that raises (not JSON, missing keys):              judge_error, nl 0, judged 0;
  - a reply whose number of results differs from the number of assertions, INCLUDING an empty or missing 'results'
    list (tau2 itself would score that all([]) = 1.0, evaluator_nl_assertions.py:134 and :51-52):  judge_error,
    nl 0, judged 0 (review issue 11);
  - a task listed as needing the judge whose task version has no assertion:  judge_error (never a silent pass);
  - otherwise nl = 1.0 iff every result is met; judged = DB x nl.
Judge errors are counted per arm. Requests pass through the campaign's replay store (a re-run is free and identical).
JSON mode is requested (response_format json_object).
--health-only (the smoke, before the freeze): writes only the judge's health (parse rate, the share of replies whose
results count equals the assertion count, seconds per call) and NO score of any kind.
Revision 3: the server identity is checked when the call starts (full) and when it ends (light: pid, /props, argv;
e4_run.identity_end_check); anything but "same" at the end exits 2 after the output is written. Revision 5
(registration 4.1, 11.6): every call first writes its start record <out>.start.json (kept as <out>.start.<k>.json at the
next call). Revision 6 (round-5 review Major 1): a judge call stopped before its end check (the operator's stop, or a
failed call) gets it AT THE STOP from run_e4_gpu.sh (scripts/end_check_at_stop.py), while the server is still up; at
the next start of the same judge run the FULL start check comes first (e4_run.start_checks: a server that is not the
pinned one exits 2 and leaves the record untouched), and only then does a start record still without its end (a kill
-9 or a crash of the script) get the deferred check: "identity_changed" exits 2 with "(identity_changed)" (that arm's
judged readings are then "not available", registration 9.1).
Run: e4 python -I -B judge_posthoc.py --results runs/<c>/<arm>/results.json --base http://127.0.0.1:18082/v1 \
       --campaign <c> --expect-server pins/server_expected.json --out runs/<c>/<arm>/judged.json [--health-only]
"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))

import argparse
import json
import sys
import time
from pathlib import Path

E4 = Path(f"{_REPO_ROOT}/studies/agent_guards/e4")
sys.path.insert(0, str(E4))
sys.path.insert(1, str(E4 / "scripts"))

STOPS = {"agent_stop", "user_stop"}


def _term(sim) -> str:
    t = sim.termination_reason
    return str(getattr(t, "value", t))


def judge_one(task, sim, needs_judge: bool, evaluate) -> dict:
    """One simulation -> its judged row. `evaluate(trajectory, assertions)` returns a list of checks with `.met`."""
    db = sim.reward_info.reward if sim.reward_info is not None else 0.0
    r = {"task": str(sim.task_id), "db": float(db), "needs_judge": bool(needs_judge)}
    early = _term(sim) not in STOPS
    if not needs_judge:
        r.update({"nl": 1.0, "judged": 0.0 if early else float(db), "note": "no NL assertions"})
        return r
    if early:
        r.update({"nl": None, "judged": 0.0, "note": "ended early; not judged"})
        return r
    assertions = list(task.evaluation_criteria.nl_assertions or [])
    if not assertions:      # listed in judge_tasks.json but no assertion in this task version: never a silent pass
        r.update({"nl": 0.0, "judged": 0.0, "judge_error": "listed as needing the judge but has no NL assertions"})
        return r
    t0 = time.perf_counter()
    try:
        checks = evaluate(sim.messages, assertions)
    except Exception as e:  # noqa: BLE001
        r.update({"nl": 0.0, "judged": 0.0, "judge_error": repr(e)[:300], "parsed": False,
                  "seconds": round(time.perf_counter() - t0, 3)})
        return r
    r["seconds"] = round(time.perf_counter() - t0, 3)
    r["parsed"] = True
    r["n_assertions"], r["n_results"] = len(assertions), len(checks)
    r["count_match"] = len(checks) == len(assertions)
    if not r["count_match"]:
        r.update({"nl": 0.0, "judged": 0.0,
                  "judge_error": f"results count {len(checks)} != assertions count {len(assertions)}"})
        return r
    nl = 1.0 if all(bool(c.met) for c in checks) else 0.0
    r.update({"nl": nl, "judged": float(db) * nl, "checks": [{"met": bool(c.met)} for c in checks]})
    return r


def judge_rows(sims, tasks: dict, need: set, evaluate) -> list[dict]:
    return [judge_one(tasks[str(s.task_id)], s, str(s.task_id) in need, evaluate) for s in sims]


def summarize(rows: list[dict], health_only: bool) -> dict:
    sent = [r for r in rows if "parsed" in r]
    h = {"judge_calls": len(sent), "parsed": sum(1 for r in sent if r["parsed"]),
         "count_match": sum(1 for r in sent if r.get("count_match")),
         "judge_errors": sum(1 for r in rows if "judge_error" in r),
         "seconds_per_call": [r["seconds"] for r in sent]}
    h["parse_rate"] = h["parsed"] / h["judge_calls"] if h["judge_calls"] else None
    h["count_match_share"] = h["count_match"] / h["judge_calls"] if h["judge_calls"] else None
    if health_only:
        return {"health": h}
    n = max(1, len(rows))
    return {"health": h, "n": len(rows), "pass1_db": sum(r["db"] for r in rows) / n,
            "pass1_judged": sum(r["judged"] for r in rows) / n, "judge_errors": h["judge_errors"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--expect-server", required=True)
    ap.add_argument("--judge-tasks", default=str(E4 / "results/judge_tasks.json"))
    ap.add_argument("--health-only", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from e4_run import die, identity_end_check, local_only_env, server_health, start_checks

    info = server_health(a.base)
    model = a.model or info["models"][0]
    pins = json.loads(Path(a.expect_server).read_text())
    stem = a.out[:-len(".json")] if a.out.endswith(".json") else a.out
    sp = Path(stem + ".start.json")                                # revision 5: this judge run's start record
    ident = start_checks(a.base, pins, sp, "judge run")            # revision 6: the full start check first
    if sp.exists():                                                # keep the earlier call's start record
        k = 1
        while Path(f"{stem}.start.{k}.json").exists():
            k += 1
        sp.rename(f"{stem}.start.{k}.json")
    sp.write_text(json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "out": a.out,
                              "server_identity": ident}, indent=1, default=str))
    local_only_env(a.base)
    from tau2.data_model.simulation import Results
    from tau2.evaluator import evaluator_nl_assertions as nl
    from tau2.registry import registry

    from e4guard.replay import ReplayStore, install

    nl.DEFAULT_LLM_NL_ASSERTIONS = "openai/" + model
    nl.DEFAULT_LLM_NL_ASSERTIONS_ARGS = {"api_base": a.base, "api_key": "sk-local-no-key", "temperature": 0.0,
                                         "max_tokens": 1024, "timeout": 600, "response_format": {"type": "json_object"},
                                         "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    store = ReplayStore(str(E4 / "runs" / a.campaign / "replay.sqlite"))
    store.arm = "judge"
    install(store)
    need = set(json.loads(Path(a.judge_tasks).read_text())["summary"]["judge_tasks_static"]["ids"])
    res = Results.load(Path(a.results))
    tasks = {str(t.id): t for t in registry.get_tasks_loader("retail")(None)}
    t0 = time.time()
    rows = judge_rows(res.simulations, tasks, need, nl.NLAssertionsEvaluator.evaluate_nl_assertions)
    s = summarize(rows, a.health_only)
    out = {"results": a.results, "model": model, "self_judged": True, "validated": False, "server_identity": ident,
           **s, "store": store.stats, "seconds": round(time.time() - t0, 1)}
    if a.health_only:
        out["rows"] = [{k: r[k] for k in ("task", "needs_judge", "parsed", "count_match", "seconds") if k in r}
                       for r in rows]
    else:
        out["rows"] = rows
    end, kind = identity_end_check(a.base, pins, ident)        # revision 3: the identity when the call ends
    out["server_identity_end"] = end
    Path(a.out).write_text(json.dumps(out, indent=1, default=str))
    rec = json.loads(sp.read_text())
    rec["server_identity_end"] = end
    sp.write_text(json.dumps(rec, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k not in ("rows", "server_identity", "server_identity_end")},
                     indent=1, default=str))
    if kind != "same":
        die(2, f"the server identity changed during the judge call ({kind}): " + "; ".join(end["differences"]))


if __name__ == "__main__":
    main()
