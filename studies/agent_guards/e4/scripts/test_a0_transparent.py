"""Offline test: GuardedAgent in monitor mode (A0) is transparent. With the same (fake, deterministic) LLM it sends
byte-identical requests to stock tau2 LLMAgent, executes the same calls in the same order and ends in the same
database, while it splits multi-call messages into one call per message. No LLM, no server.

The fake LLM replaces `generate` in both agents' modules. It is a pure function of the request: it counts the
assistant messages in the request (the k-th model reply) and returns the k-th message of a fixed plan built from the
task's gold plan, where consecutive calls are BUNDLED into one model message (so the split is exercised).

Run: offline_env.sh .../taubench/.venv/bin/python -I -B test_a0_transparent.py --out results/a0_transparent.json
"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, f"{_REPO_ROOT}/studies/agent_guards/e4")


def build_model_messages(plan):
    """Group the plan: consecutive call steps form ONE model message (max 3 calls), a text step its own message."""
    msgs, cur = [], []
    for st in plan:
        if st[0] == "call":
            cur.append(st)
            if len(cur) == 3:
                msgs.append(cur)
                cur = []
        else:
            if cur:
                msgs.append(cur)
                cur = []
            msgs.append([st])
    if cur:
        msgs.append(cur)
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default=None)
    args = ap.parse_args()
    t0 = time.time()

    from e4guard.episodes import quiet_logs

    quiet_logs()
    import tau2.agent.llm_agent as la
    from tau2.agent.llm_agent import LLMAgent
    from tau2.data_model.message import AssistantMessage, ToolCall
    from tau2.evaluator.evaluator import EvaluationType
    from tau2.orchestrator.orchestrator import Orchestrator
    from tau2.registry import registry
    from tau2.runner.helpers import load_tasks
    from tau2.runner.simulation import run_simulation
    from tau2.utils.llm_utils import to_litellm_messages

    import e4guard.agent as ga
    from e4guard.scripted import gold_plan, make_scripted_user

    state = {"plan_msgs": None, "hashes": None}

    def fake_generate(model=None, messages=None, tools=None, tool_choice=None, call_name=None, **kw):
        lm = to_litellm_messages(messages)
        req = {"model": model, "messages": lm, "tools": [t.openai_schema for t in tools] if tools else None,
               "kw": {k: v for k, v in kw.items() if k != "num_retries"}}
        state["hashes"].append(hashlib.sha256(json.dumps(req, sort_keys=True, default=str).encode()).hexdigest())
        k = sum(1 for m in lm if m["role"] == "assistant") - 1     # minus the fixed greeting
        pm = state["plan_msgs"]
        if k >= len(pm):
            return AssistantMessage(role="assistant", content="FINAL: done.", cost=0.0)
        steps = pm[k]
        if steps[0][0] == "text":
            return AssistantMessage(role="assistant", content=steps[0][1], cost=0.0)
        return AssistantMessage(role="assistant", content=None, cost=0.0, tool_calls=[
            ToolCall(id=f"m{k}c{j}", name=s[1], arguments=json.loads(json.dumps(s[2]))) for j, s in enumerate(steps)])

    la.generate = fake_generate
    ga.generate = fake_generate

    tasks = load_tasks("retail", "base")
    if args.tasks:
        keep = set(args.tasks.split(","))
        tasks = [t for t in tasks if t.id in keep]
    rows, n_ok = [], 0
    for t in tasks:
        g = gold_plan(t)
        res = {}
        for kind in ("stock", "a0"):
            state["plan_msgs"] = build_model_messages(g["plan"])
            state["hashes"] = []
            env = registry.get_env_constructor("retail")()
            if kind == "stock":
                agent = LLMAgent(tools=env.get_tools(), domain_policy=env.get_policy(), llm="fake", llm_args={})
            else:
                agent = ga.GuardedAgent(tools=env.get_tools(), domain_policy=env.get_policy(), llm="fake", llm_args={},
                                        arm="A0", task_id=t.id)
            user = make_scripted_user(g["first_message"])
            orch = Orchestrator(domain="retail", agent=agent, user=user, environment=env, task=t, max_steps=200,
                                max_errors=10, seed=300)
            sim = run_simulation(orch, evaluation_type=EvaluationType.ENV)
            calls = [(tc.name, json.dumps(tc.arguments, sort_keys=True)) for m in sim.messages
                     if getattr(m, "role", "") == "assistant" and m.tool_calls for tc in m.tool_calls]
            n_multi = sum(1 for m in sim.messages if getattr(m, "role", "") == "assistant" and m.tool_calls
                          and len(m.tool_calls) > 1)
            res[kind] = {"hashes": list(state["hashes"]), "calls": calls, "db_hash": env.get_db_hash(),
                         "db_reward": sim.reward_info.reward, "termination": str(sim.termination_reason),
                         "multi_call_messages_in_trajectory": n_multi,
                         "a0_denied": agent.stats["denied"] if kind == "a0" else None}
        same = (res["stock"]["hashes"] == res["a0"]["hashes"] and res["stock"]["calls"] == res["a0"]["calls"]
                and res["stock"]["db_hash"] == res["a0"]["db_hash"])
        n_ok += same
        rows.append({"task": t.id, "identical": same, "n_model_calls": len(res["a0"]["hashes"]),
                     "n_calls": len(res["a0"]["calls"]),
                     "stock_multi_call_messages": res["stock"]["multi_call_messages_in_trajectory"],
                     "a0_multi_call_messages": res["a0"]["multi_call_messages_in_trajectory"],
                     "db_reward": [res["stock"]["db_reward"], res["a0"]["db_reward"]],
                     "a0_denied": res["a0"]["a0_denied"]})
    out = {"tasks": len(rows), "identical": n_ok,
           "model_calls_total": sum(r["n_model_calls"] for r in rows),
           "stock_multi_call_messages_total": sum(r["stock_multi_call_messages"] for r in rows),
           "a0_multi_call_messages_total": sum(r["a0_multi_call_messages"] for r in rows),
           "db_reward_1_both": sum(1 for r in rows if r["db_reward"] == [1.0, 1.0]),
           "a0_denied_total": sum(r["a0_denied"] or 0 for r in rows),
           "rows": rows, "seconds": round(time.time() - t0, 1)}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    sys.exit(0 if n_ok == len(rows) else 1)


if __name__ == "__main__":
    main()
