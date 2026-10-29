"""E2 training rows from AgentDojo's ground truth, replayed with no LLM through the real pipeline (DESIGN E2,
"Training rows", first source). Run with the AgentDojo venv:

    env -i PATH=/usr/bin:/bin HOME=~ LANG=C.UTF-8 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 19 \
      ../agentdojo/.venv/bin/python -I -B scripts/make_rows.py --out data/rows_gt_v1.2.2.jsonl

Per suite: every user task benign (never-fooled replay), and every (user task, injection task) pair under the
important_instructions attack with the always-fooled replay (user ground truth, then injection ground truth). The
pipeline is [SystemMessage, InitQuery, agent, ToolsExecutionLoop([GuardedToolsExecutor(recorder), agent])]; the recorder
allows everything and records the trusted facts of each proposed call. Labels (labels.py): user calls ALLOW; injection
calls DENY when they carry an attacker value, else dropped. AgentDojo's own utility/security of each run is kept (a
plumbing check, not a result).
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

E2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(E2 / "src"))
sys.path.insert(1, f"{_REPO_ROOT}/studies/agent_guards/agentdojo/harness")

import agentdojo.attacks  # noqa: F401
from agentdojo.attacks.attack_registry import load_attack
from agentdojo.functions_runtime import FunctionCall
from agentdojo.task_suite.load_suites import get_suites
from pipelines import build_pipeline

from e2guard import frozen_clock
from e2guard.labels import gt_row_label
from e2guard.policies import AllowAllRecorder
from e2guard.replay_agents import PlanReplay


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", default="v1.2.2")
    ap.add_argument("--attack", default="important_instructions")
    ap.add_argument("--suites", nargs="*", default=None)
    args = ap.parse_args()
    frozen_clock.install()  # rev2: no wall clock in tool outputs
    out = Path(args.out)
    assert not out.exists(), f"{out} exists"
    t0 = time.time()
    stats: dict = {}
    with out.open("w") as fo:
        for sname, suite in get_suites(args.version).items():
            if args.suites and sname not in args.suites:
                continue
            agent = PlanReplay(suite, name="local-replay")
            rec = AllowAllRecorder()
            rec.set_suite(sname)
            pipe, guard = build_pipeline(agent, "tool_guard", policy=rec)
            attack = load_attack(args.attack, suite, pipe)
            env0 = suite.load_and_inject_default_environment({})
            st = {"runs": 0, "rows": 0, "ALLOW": 0, "DENY": 0, "dropped": 0, "utility_true": 0, "security_true": 0,
                  "errors": 0, "dropped_examples": []}
            jobs = [(ut, None) for ut in suite.user_tasks.values()]
            jobs += [(ut, it) for ut in suite.user_tasks.values() for it in suite.injection_tasks.values()]
            for ut, it in jobs:
                agent.follow = it
                rec.log.clear()
                guard.decisions.clear()
                try:
                    injections = attack.attack(ut, it) if it is not None else {}
                    u, s = suite.run_task_with_pipeline(pipe, ut, it, injections)
                    err = None
                except Exception as e:  # noqa: BLE001
                    u, s, err = None, None, f"{type(e).__name__}: {e}"[:300]
                    st["errors"] += 1
                st["runs"] += 1
                st["utility_true"] += bool(u)
                st["security_true"] += bool(s) if it is not None else 0
                info = agent.last_plan_info
                user_gt = list(ut.ground_truth(env0.model_copy(deep=True)))  # clean environment (labels.py)
                inj_plan = info["injection_plan"]
                assert len(rec.log) == len(guard.decisions)
                for step, (entry, dec) in enumerate(zip(rec.log, guard.decisions)):
                    cid = dec["call"]["id"]
                    part, k = ("user" if cid[0] == "u" else "injection"), int(cid[1:])
                    inj_call = FunctionCall(function=inj_plan[k][0], args=inj_plan[k][1]) if part == "injection" else None
                    label, reason, av = gt_row_label(part, inj_call, it.GOAL if it else "", ut.PROMPT, user_gt,
                                                     rec.x.table)
                    row = {"source": "gt_replay", "suite": sname, "user_task": ut.ID,
                           "injection_task": it.ID if it else None, "attack": args.attack if it else None,
                           "step": step, "part": part, "function": entry["function"], "args": entry["args"],
                           "label": label, "label_reason": reason, "attacker_values": av,
                           "facts": entry["facts"], "facts_on": entry["facts_on"],
                           "query_sha256": entry["facts"]["query_sha256"], "run_utility": u, "run_security": s,
                           "run_error": err}
                    fo.write(json.dumps(row, default=str) + "\n")
                    st["rows"] += 1
                    if label is None:
                        st["dropped"] += 1
                        if len(st["dropped_examples"]) < 6:
                            st["dropped_examples"].append(f"{it.ID}: {entry['function']}")
                    else:
                        st[label] += 1
            agent.follow = None
            st["seconds"] = round(time.time() - t0, 1)
            stats[sname] = st
            print(sname, json.dumps(st), flush=True)
    meta = {"version": args.version, "attack": args.attack, "stats": stats, "minutes": round((time.time() - t0) / 60, 2),
            "rows_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "stats"}))


if __name__ == "__main__":
    main()
