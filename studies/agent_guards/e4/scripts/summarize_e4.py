"""E4 summary after the GPU stages (CPU only; no LLM): every bar of REGISTRATION_E4.md revision 6 in one table,
the readings beside them, and the integrity checks. Unit-tested on known inputs (tests/test_summary.py) and run end to
end on stand-in and dry-run data before the freeze.

Reads runs/<campaign>/<arm dir>/{results.json, judged.json, guard_log.jsonl, manifest*.json} (a "<arm dir>_rerun"
folder replaces its tasks; e4guard/runs.py). Arm dirs: A0_base, A1_base (114 tasks); A2s1_test, A2s2_test, A2s3_test
(A2 with TM seeds 1, 2, 3), A3_test, A3t_test (40 test tasks; read ONLY when the go-bar said GO).
Statistic (registered): paired differences per task (1 trial), percentile bootstrap over tasks, 10,000 resamples,
numpy default_rng(20261021), 95%: idx = rng.integers(0, n, size=(10000, n)); interval = 2.5th and 97.5th percentiles of
the resampled mean differences (paired()). Beside E4.3 and E4.5: the two-level interval (two_level(): per resample,
each arm's seeds drawn with replacement, then the tasks; same seed).
Rules (primary score = the judge-free DB pass^1; the judged score is a reading, labelled "self-judged, unvalidated"):
  E4.2  A1 - A0, all 114 tasks: met iff diff >= +3.0 points and the interval's lower end > 0;
  E4.3  A2 (seed 1) - A1, the 40 test tasks, only after the go-bar said GO: met iff diff >= +2.0 and lower end > 0.
        LOW POWER (revision 3): every E4.3 row carries its power figure (gobar.json "power"; scripts/power_e43.py);
  E4.5  A2 (seed 1) - A3, the 40 test tasks, only after GO: non-inferior iff the lower end > -5.0 points (-2 tasks);
        "not informative" when fewer than 4 test tasks differ between A2 and A3 (then no verdict either way).
E4.3 and E4.5 follow gobar.json's outcome (written by STAGE=train_guard): "go" (computed), "no-go" (a dev-only line),
"not testable" (fewer than 10 veto-clean rows of a class; the learned part was not trained; section 8), "void" (the
MILP check, IC2(b), failed at train_guard), or "not run" (no gobar.json). Revision 2's file without "outcome" is read
by its decision text (a decision starting "NOT TESTABLE" is "not testable").
Completeness (review issue 9): a bar is computed only when every arm of the pair holds every task of its set; an
infrastructure error or a timeout scores 0 and is counted per arm; every bar is also read without the tasks that
ended that way in either arm; E4.2 is also read without the 10 smoke tasks.
Integrity checks (outcome forced by construction; not evidence; a failure means a bug). Revision 3 (re-review item 1):
each check touches only the bars in SCOPE; a failed check voids those bars (section 9.1), the others stand; a check
that does not apply reads "not applicable" (None), never a failure:
  IC1 (was E4.1) executed calls that our code reads as breaking E4-R1..E4-R10: 0 in A1 and every test arm read;  all bars
  IC2a results/equivalence: code gates == veto clauses, 0 mismatches;                                              all bars
  IC2b the MILP files: 13/13 guarded proofs and the domain holding, for TM seeds 1, 2 and 3;                  E4.3, E4.5
  IC3/IC4 live code_rules == veto_rules on every verdict; live facts == recomputed facts on every executed call; all bars
  IC5 on every task where A1 denied nothing: A1's trajectory and final DB equal A0's;                              all bars
  IC6 on every test task where the learned part made no learned-only denial: A2/A3/A3t equal A1;             E4.3, E4.5
Revision 4 (round-3 review M1): the integrity STOPS of runs/<c>/run_e4_gpu.log (check_pins.stop_effects: exit 2
"identity_changed" when a call ends, exit 7 "pins_changed") void the bars of section 9.1's fixed table too, and make
the judged readings of a judge call they touch "not available". Every void bar is printed WITHOUT its numbers (its row,
its readings, and the pass rates of an arm that feeds only void bars are withheld), with the decision that follows it:
in <c>_icr "not shown (integrity failure)"; in <c> before runs/<c>/integrity_failure.json exists "VOID (integrity
failure)" (the record, scripts/integrity_failure.py, decides mechanically); with a record that requires the re-run
"VOID (integrity failure)", replaced by <c>_icr; with a record that does not, "not shown (integrity failure)".
integrity.stop_events lists the stops read (integrity_failure.py checks that the summary is newer than the last one).
The reading without the smoke tasks uses the freeze record's smoke_tasks (10 to 20 tasks by the extension rule).
Revision 5 (section 11.6; round-4 review M1 (c)): a stage with 3 failed attempts (check_pins.stage_attempts on the
run log) is final; a bar that is not void reads "not shown (stage failed)" exactly when one of its arms is incomplete
(the completeness rule) and that arm's stage (A0_base a0, A1_base a1, the test arms a2a3) is final, or, for E4.3 and
E4.5, when train_guard is final with no go-bar decision; a bar whose arms are complete is computed and shown as usual;
after the judge's last attempt the arms it did not judge read "judged readings not available". Revision 5 (round-4
review m4): in <c>_icr, every bar that <c>'s integrity record did not void is labelled "a reading (the bar stands
from <c>)". Revision 6 (round-5 review minor 1; section 3.3): a rule-9 re-run that is unfinished (its list,
<arm>_rerun/rerun_tasks.txt, holds tasks its results.json lacks) is used as it stands: the tasks it finished replace
their first ends and the others keep them (score 0, counted); "anomalies" lists the unfinished tasks per arm.
Run: offline_env.sh .../taubench/.venv/bin/python -I -B summarize_e4.py --campaign <c> --out runs/<c>/summary.json
     (registered campaigns: the registration's sha256 must equal the freeze record's, runs/E4_frozen.json; exit 7
     otherwise; DRYRUN_ campaigns pass their dry record with --frozen)
"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

E4 = Path(f"{_REPO_ROOT}/studies/agent_guards/e4")
sys.path.insert(0, str(E4))
sys.path.insert(1, str(E4 / "scripts"))     # run_e4_gpu.sh runs this with python -I, which leaves scripts/ off sys.path

import numpy as np  # noqa: E402

from check_pins import ATTEMPTS_MAX, FROZEN, frozen_differences, stage_attempts, stop_effects  # noqa: E402
from e4guard import bits as BITS  # noqa: E402
from e4guard import rules  # noqa: E402
from e4guard.facts import WRITE_TOOLS, canon_args, events_from_message_dicts, replay_facts  # noqa: E402
from e4guard.models import facts_sha256  # noqa: E402
from e4guard.runs import db_reward, end_class, guard_events, load_arm, termination, trajectory_sha256  # noqa: E402

B_RESAMPLES, B_SEED = 10_000, 20261021
SMOKE_TASKS = ["0", "1", "2", "3", "4", "6", "7", "8", "10", "11"]
TEST_ARMS = ("A2s1_test", "A2s2_test", "A2s3_test", "A3_test", "A3t_test")
ARMS = ("A0_base", "A1_base") + TEST_ARMS
E42_MIN, E43_MIN, E45_MARGIN, E45_MIN_DIFFER = 3.0, 2.0, -5.0, 4
BARS = ("E4.2", "E4.3", "E4.5")
SCOPE = {"IC1": BARS, "IC2a": BARS, "IC2b": ("E4.3", "E4.5"), "IC3_IC4": BARS, "IC5": BARS, "IC6": ("E4.3", "E4.5")}
POWER_FILE = E4 / "results/power_e43_rev3.json"
SPLIT_FILE = (f"{_REPO_ROOT}/studies/agent_guards/taubench/data_untrusted/tau2-bench/extracted/"
              "tau2-bench-fc0055dc4e0a316c3f83133267fbd6faaa770992/data/tau2/domains/retail/split_tasks.json")


# ------------------------------------------------------------------------------------------------- the statistic
def boot_ci(d, n: int = B_RESAMPLES, seed: int = B_SEED) -> list[float]:
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    boot = d[idx].mean(1)
    return [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


def paired(a: dict, b: dict, tasks: list) -> dict:
    """a, b: {task: score in [0, 1]}; every task must be in both (KeyError otherwise). Points = 100 x score."""
    d = np.array([a[t] - b[t] for t in tasks], float) * 100.0
    return {"n_tasks": len(tasks), "diff_points": float(d.mean()), "ci95": boot_ci(d),
            "a": float(np.mean([a[t] for t in tasks]) * 100), "b": float(np.mean([b[t] for t in tasks]) * 100),
            "n_differ": int((d != 0).sum()), "a_better": int((d > 0).sum()), "b_better": int((d < 0).sum())}


def two_level(a_seeds: list[dict], b_seeds: list[dict], tasks: list, n: int = B_RESAMPLES, seed: int = B_SEED) -> dict:
    """Per resample: each arm's seeds drawn with replacement (as many as it has), then the tasks; the difference is
    the mean over the drawn tasks of (A averaged over its drawn seeds - B averaged over its drawn seeds). Points."""
    SA = np.array([[s[t] for t in tasks] for s in a_seeds], float)
    SB = np.array([[s[t] for t in tasks] for s in b_seeds], float)
    rng = np.random.default_rng(seed)
    bs = np.empty(n)
    for k in range(n):
        ia = rng.integers(0, len(SA), len(SA))
        ib = rng.integers(0, len(SB), len(SB))
        D = SA[ia].mean(0) - SB[ib].mean(0)
        idx = rng.integers(0, len(tasks), len(tasks))
        bs[k] = D[idx].mean()
    return {"n_seeds": [len(SA), len(SB)], "diff_points_all_seeds": float((SA.mean(0) - SB.mean(0)).mean() * 100),
            "ci95": [float(np.percentile(bs, 2.5) * 100), float(np.percentile(bs, 97.5) * 100)]}


def complete(arms: dict, names: list, tasks: list) -> str | None:
    """None when every named arm holds every task; else the reason the bar is not computed."""
    for nm in names:
        if nm not in arms:
            return f"arm {nm} did not run"
        miss = [t for t in tasks if t not in arms[nm]["score"]]
        if miss:
            return f"arm {nm} lacks {len(miss)} tasks: {miss[:10]}"
    return None


def bar_superiority(arms, a, b, tasks, key, min_pts) -> dict:
    why = complete(arms, [a, b], tasks)
    if why:
        return {"computed": False, "why": why}
    r = paired({t: arms[a]["score"][t][key] for t in tasks}, {t: arms[b]["score"][t][key] for t in tasks}, tasks)
    r["met"] = bool(r["diff_points"] >= min_pts and r["ci95"][0] > 0)
    r["computed"] = True
    return r


def bar_noninferior(arms, a, b, tasks, key) -> dict:
    why = complete(arms, [a, b], tasks)
    if why:
        return {"computed": False, "why": why}
    r = paired({t: arms[a]["score"][t][key] for t in tasks}, {t: arms[b]["score"][t][key] for t in tasks}, tasks)
    r["computed"] = True
    if r["n_differ"] < E45_MIN_DIFFER:
        r["verdict"] = f"not informative (only {r['n_differ']} tasks differ; the rule needs at least {E45_MIN_DIFFER})"
        r["non_inferior"] = None
    else:
        r["non_inferior"] = bool(r["ci95"][0] > E45_MARGIN)
        r["verdict"] = "non-inferior" if r["non_inferior"] else "not shown non-inferior"
    return r


def excluding_bad_ends(arms, names, tasks) -> list:
    return [t for t in tasks if all(arms[n]["score"].get(t, {}).get("end") == "normal" for n in names if n in arms)]


# ------------------------------------------------------------------------------------------------- per-arm rows
def score_rows(arm: dict, judged: dict) -> dict:
    out = {}
    for t, s in arm["sims"].items():
        j = judged.get(t)
        out[t] = {"db": db_reward(s), "judged": None if j is None else float(j["judged"]),
                  "judge_error": bool(j and "judge_error" in j), "end": end_class(s), "termination": termination(s)}
    return out


def load_judged(arm_dir: Path) -> dict:
    out = {}
    for d in (arm_dir, Path(str(arm_dir) + "_rerun")):
        p = d / "judged.json"
        if p.exists():
            out.update({str(r["task"]): r for r in json.loads(p.read_text())["rows"] if "judged" in r})
    return out


def executed_violations(sim: dict) -> list:
    out = []
    for _, e, f, _ in replay_facts(events_from_message_dicts(sim.get("messages") or [])):
        v = rules.rule_ids(f)
        if v:
            out.append({"name": e["name"], "rules": v})
    return out


def live_consistency(arm: dict, events: dict) -> dict:
    """IC3 and IC4 for one arm."""
    n_verdicts = n_code_neq_veto = 0
    n_exec = n_fact_mismatch = n_seq_mismatch = 0
    examples = []
    for t, sim in arm["sims"].items():
        ev = events.get(t, [])
        inits = [i for i, e in enumerate(ev) if e["event"] == "agent_init"]
        ev = ev[inits[-1]:] if inits else ev                     # the last tau2 attempt only
        verdicts = [e for e in ev if e["event"] == "verdict"]
        for v in verdicts:
            n_verdicts += 1
            if sorted(v["code_rules"]) != sorted(v["veto_rules"]):
                n_code_neq_veto += 1
                examples.append({"task": t, "call": v["call"]["name"], "code": v["code_rules"], "veto": v["veto_rules"]})
        allowed = [v for v in verdicts if v["allow"]]
        executed = [(e["name"], f) for _, e, f, _ in replay_facts(events_from_message_dicts(sim.get("messages") or []))]
        n_exec += len(executed)
        if len(allowed) != len(executed):
            n_seq_mismatch += 1
            examples.append({"task": t, "allowed_verdicts": len(allowed), "executed_calls": len(executed)})
            continue
        for v, (name, f) in zip(allowed, executed):
            if v["call"]["name"] != name or v.get("facts_sha256") != facts_sha256(f):
                n_fact_mismatch += 1
                if len(examples) < 10:
                    examples.append({"task": t, "call": name, "live_facts_sha256": v.get("facts_sha256")})
    return {"verdicts": n_verdicts, "code_neq_veto": n_code_neq_veto, "executed_calls": n_exec,
            "tasks_with_sequence_mismatch": n_seq_mismatch, "fact_mismatches": n_fact_mismatch,
            "ok": n_code_neq_veto == 0 and n_seq_mismatch == 0 and n_fact_mismatch == 0, "examples": examples[:10]}


def guard_counts(events: dict, sims: dict, tasks: dict) -> dict:
    """Readings per arm: denials by rule, learned-only denials, fallbacks, model calls, the strict-R3 monitor."""
    c = Counter()
    per_task_learned_only = {}
    deny_on, allow_on = Counter(), Counter()
    for t, ev in events.items():
        inits = [i for i, e in enumerate(ev) if e["event"] == "agent_init"]
        ev = ev[inits[-1]:] if inits else ev
        gold = ((tasks.get(t) or {}).get("evaluation_criteria") or {}).get("actions") or []
        lo = 0
        for e in ev:
            if e["event"] == "model_call":
                c["agent_model_calls"] += 1
            elif e["event"] == "fallback":
                c["fallbacks"] += 1
            elif e["event"] == "verdict":
                c["verdicts"] += 1
                if (e.get("monitor") or {}).get("strict_r3a"):
                    c["monitor_strict_r3a_would_deny"] += 1
                if not e["allow"]:
                    c["denied"] += 1
                    for r in e["code_rules"]:
                        c[f"denied_by_{r}"] += 1
                    if not e["code_rules"]:
                        c["learned_only_denials"] += 1
                        lo += 1
                if e.get("is_write") and not e.get("veto_clauses"):
                    is_gold = _gold_match(e["call"]["name"], e["call"].get("args"), gold)
                    key = "gold" if is_gold else "non_gold"
                    (deny_on if not e["allow"] else allow_on)[key] += 1
        per_task_learned_only[t] = lo
    vc = {k: {"proposed": deny_on[k] + allow_on[k], "denied": deny_on[k],
              "deny_rate": deny_on[k] / (deny_on[k] + allow_on[k]) if deny_on[k] + allow_on[k] else None,
              "allow_rate": allow_on[k] / (deny_on[k] + allow_on[k]) if deny_on[k] + allow_on[k] else None}
          for k in ("gold", "non_gold")}
    return {"counts": dict(c), "veto_clean_writes": vc, "learned_only_by_task": per_task_learned_only}


def _norm(v):
    return sorted(str(x) for x in v) if isinstance(v, list) else str(v)


def _gold_match(name, args, gold):
    for a in gold:
        ga = a.get("arguments") or {}
        if a.get("name") == name and set(ga) == set(args or {}) and all(_norm(ga[k]) == _norm(args[k]) for k in ga):
            return True
    return False


def db_hash(sim: dict, task: dict) -> str | None:
    """The final agent DB hash after replaying the trajectory (as tau2's EnvironmentEvaluator builds it)."""
    from tau2.data_model.simulation import SimulationRun
    from tau2.data_model.tasks import Task
    from tau2.registry import registry

    if not sim.get("messages"):
        return None
    s, tk = SimulationRun.model_validate(sim), Task.model_validate(task)
    ist = tk.initial_state
    env = registry.get_env_constructor("retail")()
    env.set_state(initialization_data=ist.initialization_data if ist else None,
                  initialization_actions=ist.initialization_actions if ist else None,
                  message_history=list(s.messages))
    return env.get_db_hash()


def same_as(arm_x: dict, arm_y: dict, tasks: list) -> dict:
    """IC5 / IC6: on these tasks, the trajectories (canonical) and the final DB of x equal y's. Tasks where either
    arm ended with an infrastructure error or a timeout are left out (they are counted per arm, rule 9)."""
    bad_traj, bad_db, n = [], [], 0
    for t in tasks:
        if t not in arm_x["sims"] or t not in arm_y["sims"]:
            continue
        if end_class(arm_x["sims"][t]) != "normal" or end_class(arm_y["sims"][t]) != "normal":
            continue
        n += 1
        sx, sy = arm_x["sims"][t], arm_y["sims"][t]
        if trajectory_sha256(sx) != trajectory_sha256(sy):
            bad_traj.append(t)
        task = arm_x["tasks"].get(t) or arm_y["tasks"].get(t)
        if task is not None and db_hash(sx, task) != db_hash(sy, task):
            bad_db.append(t)
    return {"tasks_checked": n, "trajectory_differs": bad_traj, "db_differs": bad_db,
            "ok": not bad_traj and not bad_db}


# ------------------------------------------------------------------------------------------------- main
def gate_outcome(gobar: dict | None) -> str:
    """'go', 'no-go', 'not testable', 'void' or 'not run' (revision 3; re-review item 1)."""
    if not gobar:
        return "not run"
    o = gobar.get("outcome")
    if o == "go":
        return "go" if gobar.get("GO") else "no-go"   # an inconsistent file never counts as GO
    if o in ("no-go", "not testable", "void"):
        return o
    if str(gobar.get("decision") or "").strip().upper().startswith("NOT TESTABLE"):
        return "not testable"                          # revision 2's file (the review's reproduced case)
    return "go" if gobar.get("GO") else "no-go"


def ic2b(milp: dict, outcome: str) -> dict:
    """IC2(b): every trained TM seed (1, 2, 3) has a MILP file with 13/13 guarded proofs and the domain holding."""
    if outcome == "not testable":
        return {"ok": None, "why": "not applicable: no learned part was trained (E4.3 not testable, section 8)"}
    if outcome == "not run" and not milp:
        return {"ok": None, "why": "not applicable: no MILP check ran (no go-bar decision)"}
    per = {}
    for k in (1, 2, 3):
        v = milp.get(f"milp_tm_seed{k}.json")
        if v is None:
            per[str(k)] = {"ok": False, "why": "missing"}
            continue
        proved = ((v.get("summary") or {}).get("guarded") or {}).get("proved")
        dom = v.get("pre_D_domain_holds")
        per[str(k)] = {"guarded_proved": proved, "domain_holds": dom,
                       "ok": bool(proved == len(BITS.VETO) and isinstance(dom, list) and len(dom) == 2
                                  and dom[0] == dom[1])}
    return {"ok": all(x["ok"] for x in per.values()), "per_seed": per}


def server_identity_reading(arm_dir: Path) -> dict:
    """Every e4_run.py call's identity records in an arm folder (manifest.json and the kept manifest.<k>.json)."""
    out = {"calls": 0, "pids": [], "end_kinds": []}
    for d in (arm_dir, Path(str(arm_dir) + "_rerun")):
        for m in sorted(Path(d).glob("manifest*.json")) if Path(d).exists() else []:
            j = json.loads(m.read_text())
            out["calls"] += 1
            for pr in (j.get("server_identity") or {}).get("processes") or []:
                out["pids"].append(pr.get("pid"))
            out["end_kinds"].append((j.get("server_identity_end") or {}).get("kind"))
    out["pids"] = sorted(set(x for x in out["pids"] if x is not None))
    return out


def _power_row(gobar: dict | None, power_fixed: dict | None) -> dict:
    pw = (gobar or {}).get("power") or {}
    fx = power_fixed or {}
    return {"low_power": True,
            "at_observed_go_bar_rates_tm_seed1": (pw.get("at_observed_rates") or {}).get("tm_seed1"),
            "at_go_bar_corner_G8_L4": pw.get("at_go_bar_corner_G8_L4", fx.get("at_go_bar_corner_G8_L4")),
            "net_points_for_80pct_power_nothing_against": (((pw.get("effect_for_80pct_power")
                                                             or fx.get("effect_for_80pct_power") or [{}])[0])
                                                           .get("net_points")),
            "source": "gobar.json (STAGE=train_guard)" if pw else "results/power_e43_rev3.json (fixed figures only)"}


ARM_FEEDS = {"A0_base": BARS, "A1_base": BARS, **{nm: ("E4.3", "E4.5") for nm in TEST_ARMS}}
ARM_STAGE = {"A0_base": "a0", "A1_base": "a1", **{nm: "a2a3" for nm in TEST_ARMS}}   # the stage that runs each arm
BAR_ARMS = {"E4.2": ("A1_base", "A0_base"), "E4.3": ("A2s1_test", "A1_base"), "E4.5": ("A2s1_test", "A3_test")}
BAR_READINGS = {"E4.2": ("E4.2_",), "E4.3": ("E4.3_", "A2_vs_tree_db"), "E4.5": ("E4.5_", "A2_vs_tree_db")}
NUMBERLESS = ("claim", "computed", "power")      # what a void bar keeps (E4.3's power figure is no result)


def _void_entry(v: dict, outcome: str, flag: str) -> dict:
    """A void bar without its numbers (revision 4; round-3 review M1 (d))."""
    out = {k: v[k] for k in NUMBERLESS if k in v}
    if isinstance(out.get("power"), dict):
        out["power"] = {**out["power"], "at_observed_go_bar_rates_tm_seed1": None,
                        "withheld": "the observed go-bar rates of a void campaign are withheld"}
    out.update({"void": True, "void_outcome": outcome, "flag": flag, "numbers_withheld": True})
    return out


def rerun_unfinished(arm_dir: Path) -> list:
    """Revision 6 (round-5 review minor 1; section 3.3): the tasks of an arm's rule-9 re-run list
    (<arm>_rerun/rerun_tasks.txt, written when the re-run starts) that its results.json does not hold. Those tasks
    keep their first infra or timeout ends (load_arm replaces only the tasks the re-run finished). [] without a list."""
    rr = Path(str(arm_dir) + "_rerun")
    lp = rr / "rerun_tasks.txt"
    if not lp.is_file():
        return []
    listed = lp.read_text().split()
    done = set()
    if (rr / "results.json").is_file():
        done = {str(x["task_id"]) for x in json.loads((rr / "results.json").read_text()).get("simulations", [])}
    return [t for t in listed if t not in done]


def incomplete_arms(arms: dict, names, tasks: list) -> list:
    """The named arms that do not hold every task (the completeness rule of section 3.3)."""
    return [nm for nm in names if nm not in arms or any(t not in arms[nm]["score"] for t in tasks)]


def stage_failed_bars(b: dict, arms: dict, base: list, test: list, outcome: str, final: set) -> None:
    """Revision 5 (section 11.6; round-4 review M1 (c)): a bar that is not computed reads "not shown (stage failed)"
    exactly when an arm it needs is incomplete and the stage that runs that arm is final (3 failed attempts); for E4.3
    and E4.5 also when train_guard is final and no go-bar decision exists. Other bars are left as they are."""
    for bar, names in BAR_ARMS.items():
        v = b.get(bar) or {}
        if v.get("computed"):
            continue
        if bar != "E4.2" and outcome == "not run" and "train_guard" in final:
            v["why"] = (f"not shown (stage failed): train_guard failed its {ATTEMPTS_MAX} attempts (section 11.6); "
                        "no go-bar decision; A2, A3, A3t were not run")
            v["stage_failed"] = ["train_guard"]
            continue
        if bar != "E4.2" and outcome != "go":
            continue
        bad = incomplete_arms(arms, names, base if bar == "E4.2" else test)
        st = sorted({ARM_STAGE[nm] for nm in bad if ARM_STAGE[nm] in final})
        if st:
            v["why"] = (f"not shown (stage failed): {', '.join(st)} failed its {ATTEMPTS_MAX} attempts (section 11.6); "
                        f"{v.get('why')}")
            v["stage_failed"] = st


def summarize(camp: Path, split: dict, smoke_tasks: list, gobar: dict | None, equivalence: dict | None,
              milp: dict, base_tasks: list | None = None, test_tasks: list | None = None,
              power_fixed: dict | None = None, log_text: str = "", record: dict | None = None,
              base_record: dict | None = None) -> dict:
    base = [str(t) for t in (base_tasks or split["base"])]
    test = [str(t) for t in (test_tasks or split["test"])]
    outcome = gate_outcome(gobar)
    se = stop_effects(log_text)                  # revision 4: the integrity stops of run_e4_gpu.log (section 9.1)
    sa = stage_attempts(log_text)                # revision 5: the failed attempts of every stage (section 11.6)
    final = set(sa["final"])
    names = ARMS if outcome == "go" else ("A0_base", "A1_base")
    out = {"campaign": camp.name, "arms": {}, "bars": {}, "readings": {}, "integrity": {}, "anomalies": [],
           "go_bar_outcome": outcome}
    if outcome != "go":
        present = [nm for nm in TEST_ARMS if (camp / nm / "results.json").exists()]
        if present:
            out["anomalies"].append(f"test-arm folders present although the go-bar outcome is {outcome!r}: {present}; "
                                    "not read")
    for nm in names:                             # revision 6 (round-5 review minor 1): an unfinished rule-9 re-run
        miss = rerun_unfinished(camp / nm)
        if miss:
            key = "rerun_test" if nm in TEST_ARMS else "rerun"
            out["anomalies"].append(
                f"the rule-9 re-run of {nm} is unfinished: tasks {miss} of its list keep their first infra or timeout "
                "ends (score 0, counted; section 3.3); "
                + (f"the re-run stage failed its {ATTEMPTS_MAX} attempts (final; section 11.6)" if key in final
                   else "its stage is not final: the re-run resumes on its list (section 11.3)"))
    arms = {}
    for nm in names:
        a = load_arm(camp / nm)
        if a is None:
            continue
        a["score"] = score_rows(a, load_judged(camp / nm))
        if nm in {x[: -len("_rerun")] if x.endswith("_rerun") else x for x in se["judged_not_available"]}:
            for t in a["score"]:                 # a judge call an integrity stop touched: not available (9.1)
                a["score"][t]["judged"] = None
            a["judged_not_available"] = True
        a["events"] = guard_events(camp / nm)
        arms[nm] = a
    for nm, a in arms.items():
        sc = a["score"]
        ts = list(sc)
        viol = {t: executed_violations(a["sims"][t]) for t in ts}
        gc = guard_counts(a["events"], a["sims"], a["tasks"])
        out["arms"][nm] = {
            "n": len(ts), "rerun_tasks": a["rerun_tasks"],
            "ends": dict(Counter(sc[t]["end"] for t in ts)),
            "terminations": dict(Counter(sc[t]["termination"] for t in ts)),
            "pass1_db": float(np.mean([sc[t]["db"] for t in ts])) if ts else None,
            "pass1_judged_self_judged_unvalidated": (float(np.mean([sc[t]["judged"] for t in ts]))
                                                     if ts and all(sc[t]["judged"] is not None for t in ts) else None),
            "judge_errors": sum(1 for t in ts if sc[t]["judge_error"]),
            "executed_violations_total": sum(len(v) for v in viol.values()),
            "executed_violations_by_rule": dict(Counter(r for v in viol.values() for x in v for r in x["rules"])),
            "executed_violations_per_task": (sum(len(v) for v in viol.values()) / len(ts)) if ts else None,
            "guard": gc["counts"], "veto_clean_writes": gc["veto_clean_writes"],
            "live_consistency": live_consistency(a, a["events"]),
            "server_identity": server_identity_reading(camp / nm),
        }
        if a.get("judged_not_available"):
            out["arms"][nm]["judged_readings"] = "not available (an integrity stop touched its judge call; section 9.1)"
        elif "judge" in final and not (camp / nm / "judged.json").exists():
            out["arms"][nm]["judged_readings"] = (f"not available (the judge failed its {ATTEMPTS_MAX} attempts before "
                                                  "judging this arm; section 11.6)")
        a["learned_only_by_task"] = gc["learned_only_by_task"]
    b, rd, ic = out["bars"], out["readings"], out["integrity"]
    # ---- E4.2
    b["E4.2"] = {"claim": "superiority, judge-free DB pass^1, A1 - A0, 114 tasks",
                 **bar_superiority(arms, "A1_base", "A0_base", base, "db", E42_MIN)}
    if b["E4.2"].get("computed"):
        rd["E4.2_judged_self_judged_unvalidated"] = _reading(arms, "A1_base", "A0_base", base, "judged")
        keep = excluding_bad_ends(arms, ["A1_base", "A0_base"], base)
        rd["E4.2_db_without_infra_or_timeout"] = _reading(arms, "A1_base", "A0_base", keep, "db")
        rd["E4.2_db_without_smoke_tasks"] = _reading(arms, "A1_base", "A0_base",
                                                    [t for t in base if t not in set(smoke_tasks)], "db")
    # ---- E4.3 / E4.5 by the go-bar's outcome
    tm = ((gobar or {}).get("models") or {}).get("tm_seed1", {})
    out["gobar"] = {"GO": outcome == "go", "outcome": outcome, "decision": (gobar or {}).get("decision"),
                    "G": tm.get("G"), "L": tm.get("L"), "forced": (gobar or {}).get("forced")}
    pr = _power_row(gobar, power_fixed)
    p_obs = pr["at_observed_go_bar_rates_tm_seed1"]
    p_txt = (f"E4.3's power at these rates {100 * p_obs:.1f}%" if p_obs is not None
             else "no observed rates")
    why = {"no-go": (f"NO-GO at the dev go-bar (TM seed 1: G = {tm.get('G')}, L = {tm.get('L')}; {p_txt}): a "
                     "dev-only line; A2, A3, A3t were not run on the test tasks"),
           "not testable": ("not testable: fewer than 10 veto-clean, deduplicated train rows of a class; the learned "
                            "part was not trained; A2, A3, A3t were not run (section 8)"),
           "void": ("void: integrity failure at STAGE=train_guard (IC2(b), the MILP check); A2, A3, A3t were not run "
                    "(section 9.1)"),
           "not run": "not run: no go-bar decision (STAGE=train_guard did not finish)"}
    if outcome != "go":
        b["E4.3"] = {"computed": False, "why": why[outcome], "power": pr}
        b["E4.5"] = {"computed": False, "why": why[outcome]}
    else:
        b["E4.3"] = {"claim": "superiority, judge-free DB pass^1, A2 (TM seed 1) - A1, 40 test tasks (low power)",
                     **bar_superiority(arms, "A2s1_test", "A1_base", test, "db", E43_MIN), "power": pr}
        b["E4.5"] = {"claim": "non-inferiority at -5.0 points (-2 tasks), DB, A2 (seed 1) - A3 (LR), 40 test tasks",
                     **bar_noninferior(arms, "A2s1_test", "A3_test", test, "db")}
        seeds = [nm for nm in ("A2s1_test", "A2s2_test", "A2s3_test") if nm in arms and not complete(arms, [nm], test)]
        for key, other in (("E4.3", "A1_base"), ("E4.5", "A3_test")):
            if b[key].get("computed") and len(seeds) >= 2 and not complete(arms, [other], test):
                rd[f"{key}_two_level"] = two_level([{t: arms[s]["score"][t]["db"] for t in test} for s in seeds],
                                                   [{t: arms[other]["score"][t]["db"] for t in test}], test)
            elif b[key].get("computed"):
                rd[f"{key}_two_level"] = {"computed": False, "why": ("not available (seeds cut or incomplete): seeds "
                                                                     f"available {seeds} (need 2 or more)")}
            if b[key].get("computed"):
                keep = excluding_bad_ends(arms, ["A2s1_test", other], test)
                rd[f"{key}_db_without_infra_or_timeout"] = _reading(arms, "A2s1_test", other, keep, "db")
                rd[f"{key}_judged_self_judged_unvalidated"] = _reading(arms, "A2s1_test", other, test, "judged")
                for s in seeds[1:]:
                    rd[f"{key}_db_{s}"] = _reading(arms, s, other, test, "db")
        if "A3t_test" in arms:
            rd["A2_vs_tree_db"] = _reading(arms, "A2s1_test", "A3t_test", test, "db")
        rd["E4.5_twin_operating_points"] = {nm: out["arms"][nm]["veto_clean_writes"]
                                            for nm in ("A2s1_test", "A2s2_test", "A2s3_test", "A3_test", "A3t_test")
                                            if nm in out["arms"]}
    stage_failed_bars(b, arms, base, test, outcome, final)     # revision 5 (section 11.6)
    out["stage_attempts"] = {"failed": sa["failed"], "final": sa["final"], "max": ATTEMPTS_MAX}
    # ---- integrity checks, each scoped to the bars it touches (section 9; SCOPE)
    ic["scope"] = {k: list(v) for k, v in SCOPE.items()}
    ic["IC1_executed_violations_zero"] = {nm: out["arms"][nm]["executed_violations_total"]
                                          for nm in ("A1_base",) + TEST_ARMS if nm in out["arms"]}
    ic["IC1_ok"] = (all(v == 0 for v in ic["IC1_executed_violations_zero"].values())
                    if ic["IC1_executed_violations_zero"] else None)
    ic["IC2_equivalence_mismatches"] = None if equivalence is None else equivalence.get("mismatches")
    ic["IC2a_ok"] = bool(equivalence is not None and equivalence.get("mismatches") == 0)
    ic["IC2b"] = ic2b(milp, outcome)
    ic["IC2_milp_guarded_proved"] = {k: (v.get("summary") or {}).get("guarded", {}).get("proved") for k, v in milp.items()}
    ic["IC2_ok"] = bool(ic["IC2a_ok"] and ic["IC2b"]["ok"] is not False)
    ic["IC3_IC4_live_consistency_ok"] = (all(out["arms"][nm]["live_consistency"]["ok"] for nm in out["arms"])
                                         if out["arms"] else None)
    if {"A0_base", "A1_base"} <= set(arms):
        nodeny = [t for t in base if t in arms["A1_base"]["sims"]
                  and arms["A1_base"]["learned_only_by_task"].get(t, 0) == 0
                  and not any(not e["allow"] for e in _last_attempt(arms["A1_base"]["events"].get(t, []))
                              if e["event"] == "verdict")]
        ic["IC5_A1_equals_A0_where_A1_denied_nothing"] = same_as(arms["A1_base"], arms["A0_base"], nodeny)
    ic6 = []
    for nm in TEST_ARMS:
        if nm in arms and "A1_base" in arms:
            quiet = [t for t in test if t in arms[nm]["sims"] and arms[nm]["learned_only_by_task"].get(t, 0) == 0]
            ic[f"IC6_{nm}_equals_A1_where_no_learned_denial"] = same_as(arms[nm], arms["A1_base"], quiet)
            ic6.append(ic[f"IC6_{nm}_equals_A1_where_no_learned_denial"]["ok"])
    i5 = ic.get("IC5_A1_equals_A0_where_A1_denied_nothing")
    checks = {"IC1": ic["IC1_ok"], "IC2a": ic["IC2a_ok"], "IC2b": ic["IC2b"]["ok"],
              "IC3_IC4": ic["IC3_IC4_live_consistency_ok"], "IC5": None if i5 is None else i5["ok"],
              "IC6": (all(ic6) if ic6 else None)}
    ic["checks"] = checks                       # True = holds, False = FAILED, None = not applicable
    ic["stop_events"] = se["events"]            # revision 4: the integrity stops read from run_e4_gpu.log
    ic["stop_void_bars"] = se["void_bars"]
    ic["judged_readings_not_available"] = se["judged_not_available"]
    ic["record"] = None if record is None else {k: record.get(k) for k in ("failed_checks", "void_bars", "rerun_required",
                                                                           "decision", "logged_utc")}
    ic["all_ok"] = not any(v is False for v in checks.values()) and not se["void_bars"]
    icr = camp.name.endswith("_icr")
    ic["by_bar"] = {}
    void_bars = []
    for bar in BARS:
        failed = [c for c, ok in checks.items() if ok is False and bar in SCOPE[c]]
        stops = [f"{e['name']} at {e['stage']}" for e in se["events"] if bar in e["void_bars"]]
        ic["by_bar"][bar] = {"checks": [c for c in SCOPE if bar in SCOPE[c]], "failed": failed, "stops": stops,
                             "void": bool(failed or stops)}
        if not (failed or stops) or bar not in b:
            continue
        void_bars.append(bar)
        why = ", ".join(failed + [f"stop {x}" for x in stops])
        if icr:
            res, flag = "not shown (integrity failure)", (f"INTEGRITY FAILURE ({why}) in the integrity re-run: not shown "
                                                          "(integrity failure); no claim line uses it; there is no "
                                                          "second re-run (section 9.1)")
        elif record is None:
            res, flag = "VOID (integrity failure)", (
                f"INTEGRITY FAILURE ({why}): void in this campaign; no claim line uses it; once the cause is found and "
                f"logged, scripts/integrity_failure.py writes the record whose mechanical decision follows: the one "
                f"integrity re-run {camp.name}_icr when the logged fix changes no file named in H1-H5, else 'not shown "
                "(integrity failure)' (section 9.1)")
        elif record.get("rerun_required"):
            res, flag = "VOID (integrity failure)", (
                f"INTEGRITY FAILURE ({why}): void in this campaign; replaced by the integrity re-run {camp.name}_icr "
                f"(the record logged {record.get('logged_utc')}: {record.get('decision')})")
        else:
            res, flag = "not shown (integrity failure)", (
                f"INTEGRITY FAILURE ({why}): not shown (integrity failure); no claim line uses it (the record logged "
                f"{record.get('logged_utc')}: {record.get('decision')})")
        b[bar] = _void_entry(b[bar], res, flag)
        for k in list(rd):                       # a void bar's readings are withheld with it
            if k.startswith(BAR_READINGS[bar]):
                rd[k] = {"computed": False, "why": f"withheld: {bar} is void (integrity failure, section 9.1)"}
    if record is not None and not icr and sorted(record.get("void_bars") or []) != sorted(void_bars):
        out["anomalies"].append(f"the integrity record's void bars {record.get('void_bars')} differ from this "
                                f"summary's {void_bars}")
    for nm, feeds in ARM_FEEDS.items():          # an arm that feeds only void bars: its pass rates are withheld
        if nm in out["arms"] and all(x in void_bars for x in feeds):
            out["arms"][nm].update({"pass1_db": None, "pass1_judged_self_judged_unvalidated": None,
                                    "pass_rates": "withheld: every bar this arm feeds is void (integrity failure)"})
    if "E4.3" in void_bars:
        out["gobar"].update({"G": None, "L": None, "withheld": "E4.3 is void (integrity failure)"})
    ic["void_bars"] = void_bars
    if icr and base_record is not None:          # revision 5 (round-4 review m4): the bars the base record did not void
        base_c = camp.name[: -len("_icr")]
        for bar in BARS:
            if bar in b and bar not in (base_record.get("void_bars") or []):
                b[bar].update({"reading_only": True, "stands_from": base_c})
    out["table"] = [_table_row(k, v, icr) for k, v in b.items()]
    return out


def _last_attempt(ev):
    inits = [i for i, e in enumerate(ev) if e["event"] == "agent_init"]
    return ev[inits[-1]:] if inits else ev


def _reading(arms, a, b_, tasks, key) -> dict:
    why = complete(arms, [a, b_], tasks)
    if why:
        return {"computed": False, "why": why}
    if any(arms[n]["score"][t][key] is None for n in (a, b_) for t in tasks):
        return {"computed": False, "why": f"{key} missing for some tasks"}
    if not tasks:
        return {"computed": False, "why": "no tasks"}
    r = paired({t: arms[a]["score"][t][key] for t in tasks}, {t: arms[b_]["score"][t][key] for t in tasks}, tasks)
    r["computed"] = True
    r["rule"] = "none (a reading)"
    return r


def _table_row(name, v, icr: bool = False) -> dict:
    row = {"bar": name, "computed": v.get("computed")}
    if v.get("void"):                            # revision 4: a void bar's row carries no number
        row["outcome"] = v.get("void_outcome") or ("not shown (integrity failure)" if icr else "VOID (integrity failure)")
    elif v.get("computed"):
        row.update({"diff_points": round(v["diff_points"], 2), "ci95": [round(x, 2) for x in v["ci95"]],
                    "n_tasks": v["n_tasks"], "outcome": v.get("verdict") or ("met" if v.get("met") else "not met")})
    else:
        why = str(v.get("why") or "")
        row["outcome"] = (f"not shown (not run): {why}" if why.startswith("arm ") and "did not run" in why else why)
    if v.get("flag"):
        row["flag"] = v["flag"]
    if "power" in v:
        row["power"] = v["power"]
    if v.get("reading_only"):                    # revision 5 (round-4 review m4): in <c>_icr, a bar <c> did not void
        row["outcome"] = f"a reading (the bar stands from {v['stands_from']}): {row['outcome']}"
        row["kind"] = "reading"
    return row


def main():
    from e4guard.episodes import quiet_logs

    quiet_logs("WARNING")              # tau2 logs every replayed tool call at DEBUG otherwise
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--split-file", default=SPLIT_FILE)
    ap.add_argument("--equivalence", default=str(E4 / "results/equivalence_code_vs_veto_rev2.json"))
    ap.add_argument("--base-tasks", nargs="*", default=None, help="DRYRUN_ campaigns only")
    ap.add_argument("--test-tasks", nargs="*", default=None, help="DRYRUN_ campaigns only")
    ap.add_argument("--smoke-tasks", nargs="*", default=None,
                    help="default: the freeze record's smoke_tasks (else the first 10 train tasks)")
    ap.add_argument("--registration", default=None, help="default: the freeze record's registration")
    ap.add_argument("--frozen", default=None, help="DRYRUN_ campaigns only: their dry freeze record")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dry = a.campaign.startswith("DRYRUN_")
    if (a.base_tasks or a.test_tasks) and not dry:
        sys.exit("REFUSING: task-set overrides are for DRYRUN_ campaigns only")
    if a.frozen and not dry:
        sys.exit("REFUSING: --frozen is for DRYRUN_ campaigns only (a registered campaign uses runs/E4_frozen.json)")
    fd = frozen_differences(Path(a.registration) if a.registration else None, Path(a.frozen) if a.frozen else FROZEN,
                            a.campaign)
    if fd:
        print(json.dumps({"refused": fd}, indent=1))
        sys.exit(7)
    camp = E4 / "runs" / a.campaign
    fz = json.loads((Path(a.frozen) if a.frozen else FROZEN).read_text())
    smoke = a.smoke_tasks if a.smoke_tasks is not None else [str(t) for t in (fz.get("smoke_tasks") or SMOKE_TASKS)]
    lp, rp = camp / "run_e4_gpu.log", camp / "integrity_failure.json"
    log_text = lp.read_text() if lp.is_file() else ""
    record = json.loads(rp.read_text()) if rp.is_file() else None
    bp = E4 / "runs" / a.campaign[: -len("_icr")] / "integrity_failure.json" if a.campaign.endswith("_icr") else None
    base_record = json.loads(bp.read_text()) if bp is not None and bp.is_file() else None
    sp = json.loads(Path(a.split_file).read_text())
    gobar = json.loads((camp / "gobar.json").read_text()) if (camp / "gobar.json").exists() else None
    eq = json.loads(Path(a.equivalence).read_text()) if Path(a.equivalence).exists() else None
    milp = {p.name: json.loads(p.read_text()) for p in sorted(camp.glob("milp_tm_seed*.json"))}
    pf = json.loads(POWER_FILE.read_text()) if POWER_FILE.exists() else None
    out = summarize(camp, sp, smoke, gobar, eq, milp, a.base_tasks, a.test_tasks, power_fixed=pf,
                    log_text=log_text, record=record, base_record=base_record)
    out["inputs"] = {"split_file_sha256": hashlib.sha256(Path(a.split_file).read_bytes()).hexdigest(),
                     "equivalence": a.equivalence, "milp_files": sorted(milp), "gobar": bool(gobar),
                     "power_file_sha256": hashlib.sha256(POWER_FILE.read_bytes()).hexdigest() if pf else None,
                     "frozen_record": str(a.frozen or FROZEN), "smoke_tasks": smoke,
                     "run_log_sha256": hashlib.sha256(lp.read_bytes()).hexdigest() if lp.is_file() else None,
                     "integrity_record_sha256": hashlib.sha256(rp.read_bytes()).hexdigest() if record else None,
                     "base_integrity_record_sha256": (hashlib.sha256(bp.read_bytes()).hexdigest()
                                                      if base_record is not None else None)}
    Path(a.out).write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({"table": out["table"], "integrity_checks": out["integrity"]["checks"],
                      "integrity_stops": [e["line"] for e in out["integrity"]["stop_events"]],
                      "integrity_all_ok": out["integrity"]["all_ok"], "gobar": out["gobar"],
                      "anomalies": out["anomalies"]}, indent=1, default=str))


if __name__ == "__main__":
    main()
