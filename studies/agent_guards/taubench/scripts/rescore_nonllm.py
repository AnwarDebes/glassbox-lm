"""Re-score saved tau2 trajectories on the reward parts that need NO LLM.

Parts recomputed with tau2's own evaluators: DB (end-state hash), ENV_ASSERTION, ACTION (gold calls matched),
COMMUNICATE (substring).
The NL_ASSERTION part (LLM judge) is never called; tasks whose reward needs it are reported separately.

Two uses:
  1. offline check of the reward function on real trajectories (e.g. the leaderboard files shipped in
     data/tau2/results/final): does our recomputation agree with the recorded db_check / reward?
  2. later, a deterministic judge-free score for our own local runs (results.json written by tau2 run).

Usage (never needs a server, an API key or the GPU):
  CUDA_VISIBLE_DEVICES= .venv/bin/python -I -B scripts/rescore_nonllm.py --results <results.json> \
      --domain airline [--fresh-tasks] [--limit 200] --out results/rescore_<name>.json
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--fresh-tasks", action="store_true",
                    help="score against the current task definitions instead of the ones embedded in the file")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from tau2.data_model.simulation import Results, TerminationReason
    from tau2.data_model.tasks import RewardType
    from tau2.environment.toolkit import get_tool_types
    from tau2.evaluator.evaluator_action import ActionEvaluator
    from tau2.evaluator.evaluator_communicate import CommunicateEvaluator
    from tau2.evaluator.evaluator_env import EnvironmentEvaluator
    from tau2.registry import registry

    t0 = time.time()
    res = Results.load(Path(args.results))
    tasks = {t.id: t for t in res.tasks}
    if args.fresh_tasks:
        tasks = {t.id: t for t in registry.get_tasks_loader(args.domain)(None)}
    ctor = registry.get_env_constructor(args.domain)
    env0 = ctor()
    tool_types = get_tool_types(env0.tools) if env0.tools is not None else {}
    if env0.user_tools is not None:
        tool_types.update(get_tool_types(env0.user_tools))
    sims = res.simulations[: args.limit] if args.limit else res.simulations

    c = Counter()
    rows = []
    for sim in sims:
        task = tasks.get(sim.task_id)
        if task is None or task.evaluation_criteria is None:
            c["task_missing_or_no_criteria"] += 1
            continue
        basis = set(task.evaluation_criteria.reward_basis)
        needs_judge = RewardType.NL_ASSERTION in basis and bool(task.evaluation_criteria.nl_assertions)
        rec = sim.reward_info
        rec_reward = rec.reward if rec is not None else None
        rec_db = rec.db_check.db_match if (rec is not None and rec.db_check is not None) else None
        row = {"task_id": sim.task_id, "trial": sim.trial, "termination": str(sim.termination_reason.value),
               "recorded_reward": rec_reward, "recorded_db_match": rec_db, "needs_judge": needs_judge}
        if sim.termination_reason not in {TerminationReason.AGENT_STOP, TerminationReason.USER_STOP}:
            row.update({"nonllm_reward": 0.0, "db_match": None, "note": "premature termination -> 0"})
            c["premature_termination"] += 1
        else:
            try:
                r = 1.0
                db_match = None
                if basis & {RewardType.DB, RewardType.ENV_ASSERTION}:
                    er = EnvironmentEvaluator.calculate_reward(environment_constructor=ctor, task=task,
                                                               full_trajectory=sim.messages, strict_replay=False)
                    r *= er.reward
                    db_match = er.db_check.db_match if er.db_check is not None else None
                if RewardType.ACTION in basis:
                    r *= ActionEvaluator.calculate_reward(task=task, full_trajectory=sim.messages,
                                                          tool_types=tool_types).reward
                if RewardType.COMMUNICATE in basis:
                    r *= CommunicateEvaluator.calculate_reward(task=task, full_trajectory=sim.messages).reward
                row.update({"nonllm_reward": r, "db_match": db_match})
            except Exception as e:
                row.update({"exception": repr(e)[:300]})
                c["exception"] += 1
                rows.append(row)
                continue
        # agreement checks
        if row.get("db_match") is not None and rec_db is not None:
            c["db_match_agree" if row["db_match"] == rec_db else "db_match_DISAGREE"] += 1
        if not needs_judge and rec_reward is not None and "nonllm_reward" in row:
            c["reward_agree" if abs(row["nonllm_reward"] - rec_reward) < 1e-9 else "reward_DISAGREE"] += 1
        if needs_judge:
            c["needs_judge"] += 1
        c["scored"] += 1
        rows.append(row)

    n = max(1, sum(1 for r in rows if "nonllm_reward" in r))
    summary = {
        "results_file": args.results, "domain": args.domain, "fresh_tasks": args.fresh_tasks,
        "n_simulations_read": len(sims), "counts": dict(c),
        "mean_recorded_reward": round(sum((r["recorded_reward"] or 0) for r in rows) / max(1, len(rows)), 4),
        "mean_nonllm_reward": round(sum(r.get("nonllm_reward", 0) for r in rows) / n, 4),
        "seconds": round(time.time() - t0, 1),
    }
    with open(args.out, "w") as fp:
        json.dump({"summary": summary, "rows": rows}, fp, indent=1, default=str)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
