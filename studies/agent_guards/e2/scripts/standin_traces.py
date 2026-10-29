"""STAND-IN traces (no LLM): AgentDojo TraceLogger JSONs of the scripted always-fooled replay, so that the D0-trace
code path (traces_to_rows.py, labels.agent_call_label) meets DENY cases before Qwen's real D0 traces exist. The
never-fooled stand-in traces already exist in harness_build/agentdojo/offline_runs/ (read-only). Nothing produced
here is a finding. AgentDojo venv:

    ../agentdojo/.venv/bin/python -I -B scripts/standin_traces.py --logdir data/standin_traces
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))

import argparse
import json
import sys
import time
from pathlib import Path

E2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(E2 / "src"))
sys.path.insert(1, f"{_REPO_ROOT}/studies/agent_guards/agentdojo/harness")

import agentdojo.attacks  # noqa: F401
from agentdojo.attacks.attack_registry import load_attack
from agentdojo.benchmark import run_task_with_injection_tasks
from agentdojo.task_suite.load_suites import get_suites
from pipelines import build_pipeline
from quiet_logger import QuietLogger

from e2guard import frozen_clock
from e2guard.replay_agents import PlanReplay


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--version", default="v1.2.2")
    ap.add_argument("--attack", default="important_instructions")
    args = ap.parse_args()
    frozen_clock.install()  # rev2: no wall clock in tool outputs
    logdir = Path(args.logdir)
    assert not logdir.exists(), f"{logdir} exists"
    t0 = time.time()
    res = {}
    with QuietLogger(str(logdir)) as qlog:
        for sname, suite in get_suites(args.version).items():
            agent = PlanReplay(suite, name="local-replay-always-fooled")
            pipe, _ = build_pipeline(agent, "none")
            attack = load_attack(args.attack, suite, pipe)
            n = u_ok = s_ok = 0
            for ut in suite.user_tasks.values():
                for it in suite.injection_tasks.values():
                    agent.follow = it
                    ures, sres = run_task_with_injection_tasks(suite, pipe, ut, attack, logdir, True, [it.ID],
                                                               args.version)
                    n += 1
                    u_ok += bool(ures[(ut.ID, it.ID)])
                    s_ok += bool(sres[(ut.ID, it.ID)])
            agent.follow = None
            res[sname] = {"pairs": n, "utility": u_ok, "attack_success": s_ok}
            print(sname, res[sname], flush=True)
        errors = qlog.errors
    print(json.dumps({"result": res, "logger_errors": errors[:5], "minutes": round((time.time() - t0) / 60, 2)}))


if __name__ == "__main__":
    main()
