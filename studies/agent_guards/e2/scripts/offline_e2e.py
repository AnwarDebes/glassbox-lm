"""Offline end-to-end test of every E2 arm through AgentDojo's real pipeline and scorers, with SCRIPTED agents (no LLM).
A plumbing test: it shows that each arm runs and what it blocks on scripted behaviour; it is not a result of E2.

    env -i PATH=/usr/bin:/bin HOME=~ LANG=C.UTF-8 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 19 \
      ../agentdojo/.venv/bin/python -I -B scripts/offline_e2e.py --models models/standin --qbits data/query_bits \
      --out results/offline_e2e_standin.json [--arms ...] [--suites ...]

Agents (e2guard.replay_agents.PlanReplay): benign runs use the never-fooled replay of the user task's ground truth;
attacked runs (important_instructions on every pair) use the always-fooled replay (user ground truth, then the
injection task's). D1 runs through LlamaServerLLM with the in-process fake client (e2guard.fake_llm; the filter keeps
every tool). Learned arms use, for each suite, the model trained with that suite held out.
Per arm and suite: AgentDojo's utility (benign), utility under attack and targeted ASR, the guard's blocks split by
the call's part (user ground truth = a false block on scripted behaviour; injection ground truth = a true block), and
the guard's decision time.
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

E2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(E2 / "src"))
sys.path.insert(1, f"{_REPO_ROOT}/studies/agent_guards/agentdojo/harness")

import agentdojo.attacks  # noqa: F401
from agentdojo.attacks.attack_registry import load_attack
from agentdojo.task_suite.load_suites import get_suites
from llamaserver_llm import LlamaServerLLM
from pipelines import build_pipeline

from e2guard import frozen_clock
from e2guard.facts import FactExtractor, ToolTable
from e2guard.fake_llm import FakeClient, FakeCompletions
from e2guard.policies import (
    AllowAllRecorder,
    D2Policy,
    D3Policy,
    D3RopePolicy,
    D4Policy,
    D5LRPolicy,
    D5TreePolicy,
    D6Policy,
    WOnly,
)
from e2guard.replay_agents import PlanReplay

SUITES = ["banking", "slack", "travel", "workspace"]
ARMS = ["d0_none", "d0_guard_allow", "w_only", "d2", "d3", "d3r", "d4_tm", "d5_lr", "d5_tree", "d6_rules",
        "d1_tool_filter"]


def fold_paths(models: Path, qbits: Path, prefix: str, suites=SUITES) -> dict:
    return {s: (str(models / f"{prefix}_{s}.json"), str(qbits / f"fold_{s}.json")) for s in suites}


def make_policy(arm: str, x: FactExtractor, models: Path, qbits: Path, suites=SUITES):
    if arm == "d0_guard_allow":
        return AllowAllRecorder(x)
    if arm == "w_only":
        return WOnly(x)
    if arm == "d2":
        return D2Policy(x)
    if arm == "d3":
        return D3Policy(x)
    if arm == "d3r":
        return D3RopePolicy(x)
    if arm == "d4_tm":
        return D4Policy(fold_paths(models, qbits, "d4_tm", suites), x)
    if arm == "d5_lr":
        return D5LRPolicy(fold_paths(models, qbits, "d5_lr", suites), x)
    if arm == "d5_tree":
        return D5TreePolicy(fold_paths(models, qbits, "d5_tree", suites), x)
    if arm == "d6_rules":
        return D6Policy(fold_paths(models, qbits, "d6_rules", suites), x)
    raise ValueError(arm)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--qbits", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", nargs="*", default=ARMS)
    ap.add_argument("--suites", nargs="*", default=SUITES)
    ap.add_argument("--version", default="v1.2.2")
    ap.add_argument("--attack", default="important_instructions")
    args = ap.parse_args()
    frozen_clock.install()  # rev2: no wall clock in tool outputs
    out = Path(args.out)
    assert not out.exists(), f"{out} exists"
    models, qbits = Path(args.models), Path(args.qbits)
    x = FactExtractor(ToolTable())
    t0 = time.time()
    res: dict = {"note": "SCRIPTED agents (ground-truth replays); plumbing test, not an E2 result",
                 "models": args.models, "arms": {}}
    suites = get_suites(args.version)
    for arm in args.arms:
        ra: dict = {}
        policy = None if arm in ("d0_none", "d1_tool_filter") else make_policy(arm, x, models, qbits, args.suites)
        for sname in args.suites:
            suite = suites[sname]
            fake = None
            if arm == "d1_tool_filter":
                fake = FakeCompletions(suite)
                agent = LlamaServerLLM(FakeClient(fake), "qwen3.5-9b-q8_0", name="local-fake")  # type: ignore[arg-type]
                pipe, guard = build_pipeline(agent, "tool_filter")
            else:
                agent = PlanReplay(suite, name="local-replay")
                if arm == "d0_none":
                    pipe, guard = build_pipeline(agent, "none")
                else:
                    policy.set_suite(sname)
                    policy.log.clear()
                    pipe, guard = build_pipeline(agent, "tool_guard", policy=policy)
            attack = load_attack(args.attack, suite, pipe)
            st = {"benign": [0, 0], "ua": [0, 0], "asr": [0, 0], "errors": 0, "blocks_user_part": 0,
                  "calls_user_part": 0, "blocks_injection_part": 0, "calls_injection_part": 0,
                  "per_pair": {}, "benign_per_task": {}}
            jobs = [(ut, None) for ut in suite.user_tasks.values()]
            jobs += [(ut, it) for ut in suite.user_tasks.values() for it in suite.injection_tasks.values()]
            for ut, it in jobs:
                if fake is not None:
                    fake.follow = it
                else:
                    agent.follow = it
                try:
                    inj = attack.attack(ut, it) if it is not None else {}
                    u, s = suite.run_task_with_pipeline(pipe, ut, it, inj)
                except Exception as e:  # noqa: BLE001
                    st["errors"] += 1
                    st.setdefault("error_examples", []).append(f"{ut.ID}|{it.ID if it else None}: {type(e).__name__}: {e}"[:300])
                    continue
                if it is None:
                    st["benign"][0] += bool(u)
                    st["benign"][1] += 1
                    st["benign_per_task"][ut.ID] = bool(u)
                else:
                    st["ua"][0] += bool(u)
                    st["ua"][1] += 1
                    st["asr"][0] += bool(s)
                    st["asr"][1] += 1
                    st["per_pair"][f"{ut.ID}|{it.ID}"] = [bool(u), bool(s)]
            if guard is not None:
                for d in guard.decisions:
                    cid = str(d["call"].get("id") or "")
                    part = "user" if cid.startswith("u") else ("injection" if cid.startswith("i") else "other")
                    if part in ("user", "injection"):
                        st[f"calls_{part}_part"] += 1
                        st[f"blocks_{part}_part"] += int(not d["allow"])
                guard.decisions.clear()
            if policy is not None and policy.log:
                ms = [e["ms"] for e in policy.log]
                st["guard_ms"] = {"n": len(ms), "median": round(statistics.median(ms), 3), "max": round(max(ms), 3)}
                policy.log.clear()
            if fake is not None:
                st["fake_requests"] = fake.n
                st["fake_filter_requests"] = fake.n_filter
            ra[sname] = st
            print(f"{time.strftime('%H:%M:%S', time.gmtime())}Z {arm:>15} {sname:>9} benign {st['benign']} ua {st['ua']} "
                  f"asr {st['asr']} blocks user {st['blocks_user_part']}/{st['calls_user_part']} "
                  f"inj {st['blocks_injection_part']}/{st['calls_injection_part']} err {st['errors']}", flush=True)
        pooled = {k: [sum(ra[s][k][0] for s in ra), sum(ra[s][k][1] for s in ra)] for k in ("benign", "ua", "asr")}
        pooled.update({k: sum(ra[s][k] for s in ra) for k in ("errors", "blocks_user_part", "calls_user_part",
                                                             "blocks_injection_part", "calls_injection_part")})
        res["arms"][arm] = {"suites": ra, "pooled": pooled}
        out.write_text(json.dumps(res, indent=1))
    if "d0_none" in res["arms"] and "d0_guard_allow" in res["arms"]:
        a, b = res["arms"]["d0_none"]["suites"], res["arms"]["d0_guard_allow"]["suites"]
        res["identity_d0_vs_allow_guard"] = all(a[s]["per_pair"] == b[s]["per_pair"] and
                                                a[s]["benign_per_task"] == b[s]["benign_per_task"] for s in a)
    res["minutes"] = round((time.time() - t0) / 60, 2)
    res["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "arms"}))


if __name__ == "__main__":
    main()
