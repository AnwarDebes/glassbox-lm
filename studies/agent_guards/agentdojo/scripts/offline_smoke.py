"""Offline AgentDojo 0.1.35 smoke test: no LLM, no server, no network.

    .venv/bin/python -I -B scripts/offline_smoke.py --out offline_results.json --logdir offline_runs
        [--versions v1 ... v1.2.2] [--e2e-versions v1.2.2 v1] [--tokenizer /path/tokenizer.json]

Part A, every version and suite: number of tools, user tasks, injection tasks, injection vectors and security cases
  (user x injection pairs); ground truth (GT) of each user task -> utility check; GT of each injection task -> security
  check (suite.check(check_injectable=False)); injection tasks whose GT is empty; user tasks injectable by the
  attacks' own content-block-aware method (BaseAttack.get_injection_candidates).
Part B, e2e versions: the real AgentPipeline / ToolsExecutionLoop / ToolsExecutor and AgentDojo's own scorers, with
  scripted agents (harness/replay.py) instead of a model:
  B1 benign, never-fooled replay      -> benchmark_suite_without_injections   -> Utility
  B2 important_instructions attack    -> benchmark_suite_with_injections      -> Utility under attack, Targeted ASR
  B3 always-fooled replay (user GT then injection GT), per pair via suite.run_task_with_pipeline -> utility, ASR
  B4 = B3 with the tool guard and an ORACLE allow-list (function names of the user task's GT; plumbing test only)
  B5 = B1 and B3 with the guard and AllowAll: must give exactly the B1 / B3 results
  Optional: largest prompt per user task in Qwen3.5 tokens (JSON-serialised messages + tool schemas; an estimate, not
  the server's own count), to size the server context.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))

import agentdojo.attacks  # noqa: E402,F401  (registers the attacks)
from agentdojo.agent_pipeline.agent_pipeline import DEFENSES  # noqa: E402
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement  # noqa: E402
from agentdojo.agent_pipeline.llms.openai_llm import _function_to_openai, _message_to_openai  # noqa: E402
from agentdojo.attacks.attack_registry import ATTACKS, load_attack  # noqa: E402
from agentdojo.attacks.base_attacks import get_model_name_from_pipeline  # noqa: E402
from agentdojo.benchmark import benchmark_suite_with_injections, benchmark_suite_without_injections  # noqa: E402
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime  # noqa: E402
from agentdojo.task_suite.load_suites import get_suites  # noqa: E402
from agentdojo.types import ChatMessage  # noqa: E402

from guard import AllowAll, AllowListPolicy  # noqa: E402
from pipelines import build_pipeline  # noqa: E402
from quiet_logger import QuietLogger  # noqa: E402
from replay import ReplayLLM  # noqa: E402

VERSIONS = ["v1", "v1.1", "v1.1.1", "v1.1.2", "v1.2", "v1.2.1", "v1.2.2"]
ATTACK = "important_instructions"


class SizeProbe(BasePipelineElement):
    """Wraps the model element; records the largest prompt (messages + tool schemas) per query, in tokens."""

    def __init__(self, inner: BasePipelineElement, tokenizer) -> None:
        self.inner = inner
        self.name = inner.name
        self.tok = tokenizer
        self.max_tokens: dict[str, int] = {}

    def query(self, query: str, runtime: FunctionsRuntime, env: Env = EmptyEnv(),
              messages: Sequence[ChatMessage] = [], extra_args: dict = {}):
        if self.tok is not None:
            msgs = json.dumps([_message_to_openai(m, "x") for m in messages], ensure_ascii=False)
            tools = json.dumps([_function_to_openai(t) for t in runtime.functions.values()], ensure_ascii=False)
            n = len(self.tok.encode(msgs).ids) + len(self.tok.encode(tools).ids)
            self.max_tokens[query] = max(n, self.max_tokens.get(query, 0))
        return self.inner.query(query, runtime, env, messages, extra_args)

    @property
    def follow_injection(self):
        return self.inner.follow_injection

    @follow_injection.setter
    def follow_injection(self, value):
        self.inner.follow_injection = value


def rate(values) -> float | None:
    values = list(values)
    return round(sum(values) / len(values), 4) if values else None


def part_a(version: str) -> dict:
    out = {}
    for name, suite in get_suites(version).items():
        replay = ReplayLLM(suite, name="local-replay")
        pipeline, _ = build_pipeline(replay, "none")
        attack = load_attack(ATTACK, suite, pipeline)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _, (ut_res, it_res) = suite.check(check_injectable=False)
        env0 = suite.load_and_inject_default_environment({})
        empty_gt = sorted(i for i, t in suite.injection_tasks.items() if len(t.ground_truth(env0.model_copy(deep=True))) == 0)
        injectable, not_injectable, n_vectors_seen = [], [], []
        for ut_id, ut in suite.user_tasks.items():
            try:
                cands = attack.get_injection_candidates(ut)
                injectable.append(ut_id)
                n_vectors_seen.append(len(cands))
            except ValueError:
                not_injectable.append(ut_id)
        prompts = Counter(t.PROMPT for t in suite.user_tasks.values())
        out[name] = {
            "tools": len(suite.tools),
            "user_tasks": len(suite.user_tasks),
            "injection_tasks": len(suite.injection_tasks),
            "injection_vectors": len(suite.get_injection_vector_defaults()),
            "security_cases": len(suite.user_tasks) * len(suite.injection_tasks),
            "gt_user_utility_pass": sum(r[0] for r in ut_res.values()),
            "gt_user_utility_fail": sorted(k for k, r in ut_res.items() if not r[0]),
            "gt_injection_security_pass": sum(bool(v) for v in it_res.values()),
            "gt_injection_security_fail": sorted(k for k, v in it_res.items() if not v),
            "injection_tasks_with_empty_gt": empty_gt,
            "injectable_user_tasks": len(injectable),
            "not_injectable_user_tasks": not_injectable,
            "injection_vectors_seen_per_task_min_max": [min(n_vectors_seen), max(n_vectors_seen)] if n_vectors_seen else None,
            "duplicate_prompts": sum(1 for c in prompts.values() if c > 1),
            "warnings_in_check": len(caught),
        }
    return out


def per_pair(suite, pipeline, replay, attack, pairs, guard=None, policy_for=None) -> dict:
    util, sec, errors = {}, {}, {}
    for ut, it in pairs:
        replay.follow_injection = it
        if guard is not None and policy_for is not None:
            guard.policy = policy_for(ut)
            guard.context = {"suite": suite.name, "user_task": ut.ID, "injection_task": it.ID}
        try:
            u, s = suite.run_task_with_pipeline(pipeline, ut, it, attack.attack(ut, it))
            util[(ut.ID, it.ID)], sec[(ut.ID, it.ID)] = bool(u), bool(s)
        except Exception as e:  # noqa: BLE001
            errors[f"{ut.ID}|{it.ID}"] = f"{type(e).__name__}: {e}"[:300]
    replay.follow_injection = None
    return {"util": util, "sec": sec, "errors": errors}


def part_b(version: str, logdir: Path, tokenizer) -> dict:
    out = {}
    for name, suite in get_suites(version).items():
        t0 = time.time()
        res: dict = {}
        replay = ReplayLLM(suite, name="local-replay")
        probe = SizeProbe(replay, tokenizer)
        pipeline, _ = build_pipeline(probe, "none")
        attack = load_attack(ATTACK, suite, pipeline)
        res["attack_model_name"] = get_model_name_from_pipeline(pipeline)
        injectable = []
        for ut_id, ut in suite.user_tasks.items():
            try:
                attack.get_injection_candidates(ut)
                injectable.append(ut_id)
            except ValueError:
                pass
        res["injectable_user_tasks_used"] = len(injectable)
        with QuietLogger(str(logdir / version)):
            b1 = benchmark_suite_without_injections(pipeline, suite, logdir=None, force_rerun=True,
                                                    benchmark_version=version)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                b2 = benchmark_suite_with_injections(pipeline, suite, attack, logdir=None, force_rerun=True,
                                                     user_tasks=injectable, verbose=False, benchmark_version=version)
        res["B1_benign"] = {"runs": len(b1["utility_results"]), "utility": rate(b1["utility_results"].values()),
                            "failed": sorted(k[0] for k, v in b1["utility_results"].items() if not v)}
        res["B2_attack"] = {
            "runs": len(b2["utility_results"]),
            "utility_under_attack": rate(b2["utility_results"].values()),
            "targeted_asr": rate(b2["security_results"].values()),
            "attack_success_pairs": sorted(f"{k[0]}|{k[1]}" for k, v in b2["security_results"].items() if v),
            "injection_tasks_solved_as_user_tasks": f"{sum(b2['injection_tasks_utility_results'].values())}/{len(b2['injection_tasks_utility_results'])}",
            "warnings": sorted({str(w.message)[:120] for w in caught}),
        }
        if tokenizer is not None:
            prompts = {t.PROMPT: t.ID for t in suite.user_tasks.values()}
            sizes = {prompts[q]: n for q, n in probe.max_tokens.items() if q in prompts}
            top = sorted(sizes.items(), key=lambda kv: -kv[1])
            res["prompt_tokens_estimate"] = {"max": top[0][1] if top else None, "largest_5": top[:5],
                                             "median": sorted(sizes.values())[len(sizes) // 2] if sizes else None,
                                             "smallest_5": top[-5:]}
        probe.tok = None  # token counts only for B1/B2
        pairs = [(suite.user_tasks[ut_id], it) for ut_id in injectable for it in suite.injection_tasks.values()
                 if len(it.ground_truth(suite.load_and_inject_default_environment({}))) > 0]
        b3 = per_pair(suite, pipeline, probe, attack, pairs)
        res["B3_always_fooled"] = {"pairs": len(pairs), "utility": rate(b3["util"].values()),
                                   "asr": rate(b3["sec"].values()), "errors": b3["errors"],
                                   "not_successful_pairs": sorted(f"{k[0]}|{k[1]}" for k, v in b3["sec"].items() if not v)}
        gt_functions = {ut.ID: {c.function for c in ut.ground_truth(suite.load_and_inject_default_environment({}))}
                        for ut in suite.user_tasks.values()}
        g_pipe, guard = build_pipeline(probe, "tool_guard", policy=AllowListPolicy(set(), "oracle_allow_list"))
        b4 = per_pair(suite, g_pipe, probe, attack, pairs, guard,
                      lambda ut: AllowListPolicy(gt_functions[ut.ID], "oracle_allow_list"))
        n_blocked = sum(1 for d in guard.decisions if not d["allow"])
        res["B4_always_fooled_oracle_guard"] = {
            "pairs": len(pairs), "utility": rate(b4["util"].values()), "asr": rate(b4["sec"].values()),
            "errors": b4["errors"], "decisions": len(guard.decisions), "blocked_calls": n_blocked,
            "attack_success_pairs": sorted(f"{k[0]}|{k[1]}" for k, v in b4["sec"].items() if v)[:40],
        }
        a_pipe, a_guard = build_pipeline(probe, "tool_guard", policy=AllowAll())
        with QuietLogger(str(logdir / version)):
            b5_benign = benchmark_suite_without_injections(a_pipe, suite, logdir=None, force_rerun=True,
                                                           benchmark_version=version)
        b5_fooled = per_pair(suite, a_pipe, probe, attack, pairs)
        res["B5_allow_all_guard"] = {
            "benign_identical_to_B1": {k[0]: v for k, v in b5_benign["utility_results"].items()}
            == {k[0]: v for k, v in b1["utility_results"].items()},
            "fooled_identical_to_B3": b5_fooled["util"] == b3["util"] and b5_fooled["sec"] == b3["sec"],
            "decisions": len(a_guard.decisions),
            "blocked_calls": sum(1 for d in a_guard.decisions if not d["allow"]),
        }
        res["seconds"] = round(time.time() - t0, 1)
        out[name] = res
        print(f"[{version}/{name}] B1 {res['B1_benign']['utility']} | B2 util {res['B2_attack']['utility_under_attack']}"
              f" ASR {res['B2_attack']['targeted_asr']} | B3 util {res['B3_always_fooled']['utility']}"
              f" ASR {res['B3_always_fooled']['asr']} | B4 util {res['B4_always_fooled_oracle_guard']['utility']}"
              f" ASR {res['B4_always_fooled_oracle_guard']['asr']} | B5 {res['B5_allow_all_guard']['benign_identical_to_B1']}"
              f"/{res['B5_allow_all_guard']['fooled_identical_to_B3']} | {res['seconds']} s", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--versions", nargs="*", default=VERSIONS)
    ap.add_argument("--e2e-versions", nargs="*", default=["v1.2.2", "v1"])
    ap.add_argument("--tokenizer", default=None)
    args = ap.parse_args()
    out_path = Path(args.out)
    assert not out_path.exists(), f"{out_path} exists"
    tokenizer = None
    if args.tokenizer:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(args.tokenizer)
    result = {"attacks_registered": sorted(ATTACKS), "upstream_defences": list(DEFENSES), "part_a": {}, "part_b": {}}
    for v in args.versions:
        t0 = time.time()
        result["part_a"][v] = part_a(v)
        tot = {k: sum(s[k] for s in result["part_a"][v].values())
               for k in ("user_tasks", "injection_tasks", "security_cases", "gt_user_utility_pass",
                         "gt_injection_security_pass", "injectable_user_tasks")}
        result["part_a"][v]["_total"] = tot
        print(f"[A {v}] {tot} ({time.time() - t0:.1f} s)", flush=True)
    for v in args.e2e_versions:
        result["part_b"][v] = part_b(v, Path(args.logdir), tokenizer)
    out_path.write_text(json.dumps(result, indent=1, default=str))
    print("written", out_path)


if __name__ == "__main__":
    main()
