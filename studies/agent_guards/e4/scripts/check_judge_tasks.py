"""E4 step 1: which tau2-bench v1.0.1 retail tasks need the LLM judge for their reward? (no LLM is ever called)

Two independent checks on all 114 retail tasks (split_tasks.json: train 74, test 40):
  (a) static:      NL_ASSERTION in reward_basis AND a non-empty nl_assertions list
                   (evaluator.py:215-220 calls the NL evaluator only when NL_ASSERTION is in the basis;
                    evaluator_nl_assertions.py:37-44 returns 1.0 WITHOUT any LLM call when the list is empty);
  (b) behavioural: tau2's own evaluate_simulation(EvaluationType.ALL) on a scripted gold episode of every task,
                   with the judge's `generate` (tau2.evaluator.evaluator_nl_assertions.generate) replaced by a stub
                   that records the call and returns an all-met verdict. A task "needs the judge" iff the stub was
                   called. The stub never contacts any server.
Also computes, per task, the judge-free DB part of the reward on the same gold episode (tau2's EnvironmentEvaluator)
and the reward-basis inventory, so the judge-free score definition can be checked against the basis.

Run (taubench venv, read-only; network blocked by a dead proxy):
  CUDA_VISIBLE_DEVICES= .../taubench/.venv/bin/python -I -B check_judge_tasks.py --out <json>
"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

E4 = Path(f"{_REPO_ROOT}/studies/agent_guards/e4")
sys.path.insert(0, str(E4))


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()

    from tau2.data_model.message import AssistantMessage
    from tau2.data_model.tasks import RewardType
    from tau2.evaluator import evaluator_nl_assertions as nl
    from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
    from tau2.evaluator.evaluator_env import EnvironmentEvaluator
    from tau2.registry import registry
    from tau2.runner.helpers import load_task_splits, load_tasks
    from tau2.utils.utils import DATA_DIR

    from e4guard.episodes import gold_trajectory_direct, sim_run

    calls = []

    def judge_stub(model=None, messages=None, call_name=None, **kw):  # never contacts a server
        calls.append({"model": model, "call_name": call_name})
        user_prompt = messages[-1].content if messages else ""
        # all-met verdict, so the stub cannot lower a reward; only the CALL is what we record
        n = user_prompt.count("', '") + 1 if "[" in user_prompt else 1
        return AssistantMessage(role="assistant", content=json.dumps(
            {"results": [{"expectedOutcome": f"stub {i}", "reasoning": "stub", "metExpectation": True} for i in range(n)]}))

    nl.generate = judge_stub

    dom = "retail"
    splits = load_task_splits(dom)
    tasks = load_tasks(dom, "base")
    ctor = registry.get_env_constructor(dom)
    split_of = {}
    for name in ("train", "test"):
        for tid in splits[name]:
            split_of[tid] = name

    rows, basis_counts = [], Counter()
    for task in tasks:
        ec = task.evaluation_criteria
        basis = sorted(str(b.value) for b in ec.reward_basis)
        basis_counts["+".join(basis)] += 1
        n_nl = len(ec.nl_assertions or [])
        static_needs = (RewardType.NL_ASSERTION in ec.reward_basis) and n_nl > 0
        traj, st = gold_trajectory_direct(task, ctor)
        n_before = len(calls)
        ri = evaluate_simulation(simulation=sim_run(task, traj), task=task, evaluation_type=EvaluationType.ALL,
                                 solo_mode=False, domain=dom)
        judge_called = len(calls) > n_before
        er = EnvironmentEvaluator.calculate_reward(environment_constructor=ctor, task=task, full_trajectory=traj)
        rows.append({
            "task_id": task.id, "split": split_of.get(task.id), "reward_basis": basis, "n_nl_assertions": n_nl,
            "needs_judge_static": static_needs, "judge_called": judge_called,
            "gold_reward_all_with_stub_judge": ri.reward, "gold_db_match": er.db_check.db_match,
            "gold_db_reward": er.db_check.db_reward, "gold_tool_errors": st["n_tool_errors"],
            "nl_assertions": list(ec.nl_assertions or []),
        })

    def ids(pred):
        return [r["task_id"] for r in rows if pred(r)]

    judge_static = ids(lambda r: r["needs_judge_static"])
    judge_beh = ids(lambda r: r["judge_called"])
    summary = {
        "tau2_data_dir": str(DATA_DIR),
        "files_sha256": {
            "retail/tasks.json": sha256_file(Path(DATA_DIR) / "tau2/domains/retail/tasks.json"),
            "retail/split_tasks.json": sha256_file(Path(DATA_DIR) / "tau2/domains/retail/split_tasks.json"),
            "retail/policy.md": sha256_file(Path(DATA_DIR) / "tau2/domains/retail/policy.md"),
        },
        "n_tasks": len(rows), "split_sizes": {k: len(v) for k, v in splits.items()},
        "reward_basis_counts": dict(basis_counts),
        "judge_tasks_static": {"n": len(judge_static), "train": len([t for t in judge_static if split_of.get(t) == "train"]),
                               "test": len([t for t in judge_static if split_of.get(t) == "test"]), "ids": judge_static},
        "judge_tasks_behavioural": {"n": len(judge_beh), "train": len([t for t in judge_beh if split_of.get(t) == "train"]),
                                    "test": len([t for t in judge_beh if split_of.get(t) == "test"]), "ids": judge_beh},
        "static_equals_behavioural": judge_static == judge_beh,
        "judge_stub_calls": len(calls),
        "judge_model_named_in_calls": sorted({str(c["model"]) for c in calls}),
        "basis_components_other_than_DB_and_NL": sorted({b for r in rows for b in r["reward_basis"]} - {"DB", "NL_ASSERTION"}),
        "gold_db_reward_1": sum(r["gold_db_reward"] == 1.0 for r in rows),
        "gold_reward_all_with_stub_judge_1": sum(r["gold_reward_all_with_stub_judge"] == 1.0 for r in rows),
        "nl_assertions_per_judge_task": dict(Counter(r["n_nl_assertions"] for r in rows if r["needs_judge_static"])),
        "seconds": round(time.time() - t0, 1),
    }
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
