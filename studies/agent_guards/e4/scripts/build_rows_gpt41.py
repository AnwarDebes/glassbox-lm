"""STAND-IN rows from the shipped gpt-4.1 retail trajectories (tau2-bench v1.0.1 data/tau2/results/final; 2025 runs on
the OLD embedded task versions, 4 trials x 114 tasks). Used ONLY to test the E4 code; never to claim anything, never
for a registered model. No LLM is called.

For every simulation: the trajectory -> facts (facts.py, calls taken one at a time in their execution order) ->
  - every executed call through the code gates (how often would A1 have fired on this agent's real behaviour?);
  - every executed write -> one row: bits + label (DENY iff not one of the EMBEDDED task's gold actions with matching
    arguments, lists as multisets) + task id + split (current split_tasks.json) + trial + the simulation's reward;
  - R3 detail at every executed write: is the order id listed, are the item ids listed, is the latest user message an
    explicit yes (the yes-matcher's acceptance on real confirmations).
Run: offline_env.sh .../taubench/.venv/bin/python -I -B build_rows_gpt41.py --results <file> --out-rows <json> --out-stats <json>
"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

E4 = Path(f"{_REPO_ROOT}/studies/agent_guards/e4")
sys.path.insert(0, str(E4))

from e4guard import bits as B  # noqa: E402
from e4guard import rules  # noqa: E402
from e4guard.facts import WRITE_TOOLS, canon_args, events_from_message_dicts, is_explicit_yes, replay_facts  # noqa: E402


def norm(v):
    return sorted(str(x) for x in v) if isinstance(v, list) else str(v)


def gold_match(name, args, gold):
    for a in gold:
        if a.get("name") != name:
            continue
        ga = a.get("arguments") or {}
        if set(ga) == set(args or {}) and all(norm(ga[k]) == norm(args[k]) for k in ga):
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--split-file", required=True)
    ap.add_argument("--out-rows", required=True)
    ap.add_argument("--out-stats", required=True)
    args = ap.parse_args()
    t0 = time.time()
    d = json.loads(Path(args.results).read_text())
    splits = json.loads(Path(args.split_file).read_text())
    split_of = {t: s for s in ("train", "test") for t in splits[s]}
    tasks = {t["id"]: t for t in d["tasks"]}
    rows, st = [], Counter()
    per_rule_exec, per_rule_ok_sims = Counter(), Counter()
    yes_seen, r3 = Counter(), Counter()
    denied_writes_success = Counter()
    strict_only_by_tool = Counter()
    for sim in d["simulations"]:
        task = tasks.get(sim["task_id"])
        gold = (task.get("evaluation_criteria") or {}).get("actions") or [] if task else []
        reward = (sim.get("reward_info") or {}).get("reward")
        st["simulations"] += 1
        st[f"reward_{reward}"] += 1
        ev = events_from_message_dicts(sim["messages"])
        sim_rules = set()
        for _, e, f, res in replay_facts(ev):
            st["executed_calls"] += 1
            v = rules.rule_ids(f)
            for r in v:
                per_rule_exec[r] += 1
                sim_rules.add(r)
            if e["name"] in WRITE_TOOLS:
                st["executed_writes"] += 1
                st["executed_writes_tool_error"] += int(bool((res or {}).get("error")))
                lab = 0 if gold_match(e["name"], e.get("args"), gold) else 1
                st[f"write_label_{'ALLOW' if lab == 0 else 'DENY'}"] += 1
                r3["writes"] += 1
                r3["order_id_listed"] += int(f["order_id_listed"])
                r3["strict_ids_listed_rev1"] += int(f["strict_ids_listed"])
                r3["item_ids_listed"] += int(f["items_listed"])
                r3["user_yes"] += int(f["user_yes"])
                r3["both_strict_R3_parts_rev1"] += int(f["strict_ids_listed"] and f["user_yes"])
                strict = rules.strict_r3a_fires(f)
                if reward == 1.0:
                    st["writes_in_reward1_sims"] += 1
                    if v:
                        for r in v:
                            denied_writes_success[r] += 1
                        st["writes_in_reward1_sims_A1_would_deny"] += 1
                        st["writes_in_reward1_sims_A1_would_deny_by_E4-R3"] += int("E4-R3" in v)
                    if v or strict:
                        st["writes_in_reward1_sims_rev1_strict_would_deny"] += 1
                    if strict or "E4-R3" in v:
                        st["writes_in_reward1_sims_rev1_strict_would_deny_by_R3"] += 1
                    if strict and not v:
                        st["writes_in_reward1_sims_denied_only_by_rev1_strict_R3a"] += 1
                        strict_only_by_tool[e["name"]] += 1
                x = B.encode(f)
                rows.append({"source": "gpt41_standin", "task": sim["task_id"], "split": split_of.get(sim["task_id"]),
                             "trial": sim.get("trial"), "sim_reward": reward, "tool": e["name"], "label": lab,
                             "args": json.dumps(canon_args(e.get("args") or {}), sort_keys=True, default=str),
                             "x": x.tolist(), "code_rules": v, "veto_clauses": B.veto_fires(x),
                             "veto_clean": not B.veto_fires(x), "strict_r3a": strict,
                             "tool_error": bool((res or {}).get("error"))})
        for r in sim_rules:
            per_rule_ok_sims[r] += int(reward == 1.0)
        st["sims_with_any_violation"] += int(bool(sim_rules))
        st["reward1_sims_with_any_violation"] += int(bool(sim_rules) and reward == 1.0)
        # yes-matcher on the user message right before each executed write
        last_user = None
        for e in ev:
            if e["kind"] == "user_text":
                last_user = e["text"]
            if e["kind"] == "agent_call" and e["name"] in WRITE_TOOLS:
                yes_seen["pre_write_user_messages"] += 1
                yes_seen["accepted_as_yes"] += int(is_explicit_yes(last_user))
    stats = {"stand_in": True, "results_file": args.results, "counts": dict(st),
             "executed_calls_violating_rule": dict(per_rule_exec),
             "reward1_sims_with_rule": dict(per_rule_ok_sims), "r3_at_executed_writes": dict(r3),
             "yes_matcher_on_pre_write_user_messages": dict(yes_seen),
             "writes_in_reward1_sims_denied_by_rule": dict(denied_writes_success),
             "writes_in_reward1_sims_denied_only_by_rev1_strict_R3a_by_tool": dict(strict_only_by_tool),
             "r3_reading": "enforced = text-literal E4-R3 (revision 2); rev1_strict = monitor only",
             "seconds": round(time.time() - t0, 1)}
    Path(args.out_rows).write_text(json.dumps({"names": B.NAMES, "stand_in": True, "rows": rows}))
    Path(args.out_stats).write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
