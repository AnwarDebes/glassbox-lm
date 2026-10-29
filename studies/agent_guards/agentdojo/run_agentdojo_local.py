"""AgentDojo 0.1.35 against a local OpenAI-compatible llama-server (Qwen3.5-9B Q8_0), with optional tool guard.

FOR THE GPU SLOT. Normally started by run_gpu_slot.sh, which starts and stops the server. It never starts a server.

    .venv/bin/python -I -B run_agentdojo_local.py --base-url http://127.0.0.1:18081/v1 --model qwen3.5-9b-q8_0 \
        --out runs_gpu/<stamp>/agentdojo [--subset small|full|<file.json>] [--defences none tool_guard ...] \
        [--guard-policy allow_all | module:/path/policy.py:ClassName] [--dry-run]

Per defence it runs (1) every selected user task without attack -> Utility, and (2) every selected
(user task, injection task) pair with the attack (default important_instructions) -> Utility under attack and
Targeted ASR. It calls AgentDojo's own per-task runners (agentdojo.benchmark.run_task_without_injection_tasks /
run_task_with_injection_tasks), so scoring and the per-run JSON traces are upstream's; each run is wrapped in
try/except so one failure does not end the whole run. Errors are counted separately (never folded in silently):
summary.json gives rates over completed runs and a worst-case variant (error = utility 0, attack success 1, the
rule upstream uses for context-length errors, benchmark.py:120-130).

--dry-run builds the client, model element, pipelines and attacks, checks the attack can name the model and that every
selected user task is injectable, prints the plan and exits. It sends no request to any server and writes no files.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
import traceback
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "harness"))

import openai  # noqa: E402

import agentdojo.attacks  # noqa: E402,F401  (registers the attacks)
from agentdojo.attacks.attack_registry import load_attack  # noqa: E402
from agentdojo.attacks.base_attacks import get_model_name_from_pipeline  # noqa: E402
from agentdojo.benchmark import run_task_with_injection_tasks, run_task_without_injection_tasks  # noqa: E402
from agentdojo.task_suite.load_suites import get_suite  # noqa: E402

from guard import AllowAll  # noqa: E402
from llamaserver_llm import LlamaServerLLM  # noqa: E402
from pipelines import build_pipeline  # noqa: E402
from quiet_logger import QuietLogger  # noqa: E402

# Small first subset for v1.2.2 (2 user tasks x 2 injection tasks per suite = 16 attacked pairs, 8 benign runs per
# defence). User tasks: the two injectable ones with the smallest largest-prompt in the offline estimate
# (offline_results.json, part_b.v1.2.2.<suite>.prompt_tokens_estimate.smallest_5; Qwen3.5 tokens of the JSON messages +
# tool schemas in the ground-truth replay): workspace 0 (4,136) and 3 (4,145); travel 0 (4,750) and 13 (4,937);
# banking 13 (2,011) and 0 (2,050); slack 0 (1,535) and 3 (1,654). Injection tasks: the first two IDs of each suite
# (slack IDs start at 1).
SUBSETS: dict[str, dict[str, dict[str, list[str]]]] = {
    "small": {
        "banking": {"user_tasks": ["user_task_13", "user_task_0"], "injection_tasks": ["injection_task_0", "injection_task_1"]},
        "slack": {"user_tasks": ["user_task_0", "user_task_3"], "injection_tasks": ["injection_task_1", "injection_task_2"]},
        "travel": {"user_tasks": ["user_task_0", "user_task_13"], "injection_tasks": ["injection_task_0", "injection_task_1"]},
        "workspace": {"user_tasks": ["user_task_0", "user_task_3"], "injection_tasks": ["injection_task_0", "injection_task_1"]},
    },
}


def load_policy(spec: str, kwargs: dict):
    if spec == "allow_all":
        return AllowAll()
    if spec.startswith("module:"):
        _, path, cls_name = spec.split(":", 2)
        mod_spec = importlib.util.spec_from_file_location("guard_policy_module", path)
        assert mod_spec is not None and mod_spec.loader is not None, path
        module = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(module)
        return getattr(module, cls_name)(**kwargs)
    raise ValueError(f"unknown --guard-policy {spec!r}")


def select(version: str, subset: str, suites: list[str]) -> dict:
    if subset == "full":
        chosen = {}
        for s in suites:
            suite = get_suite(version, s)
            chosen[s] = {"user_tasks": list(suite.user_tasks), "injection_tasks": list(suite.injection_tasks)}
        return chosen
    table = SUBSETS[subset] if subset in SUBSETS else json.loads(Path(subset).read_text())
    return {s: table[s] for s in suites if s in table}


def http_json(url: str, timeout: float = 10.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def summarise(rows: list[dict]) -> dict:
    out: dict = {}
    for defence in sorted({r["defence"] for r in rows}):
        d: dict = {}
        for scope in ["ALL", *sorted({r["suite"] for r in rows})]:
            rs = [r for r in rows if r["defence"] == defence and (scope == "ALL" or r["suite"] == scope)]
            ben = [r for r in rs if r["attack"] is None]
            att = [r for r in rs if r["attack"] is not None]

            def rate(xs, key, worst):
                ok = [x for x in xs if x["error"] is None]
                done = round(sum(bool(x[key]) for x in ok) / len(ok), 4) if ok else None
                worst_rate = round(sum((worst if x["error"] is not None else bool(x[key])) for x in xs) / len(xs), 4) if xs else None
                return done, worst_rate

            u, u_w = rate(ben, "utility", False)
            ua, ua_w = rate(att, "utility", False)
            asr, asr_w = rate(att, "security", True)
            d[scope] = {
                "benign_runs": len(ben), "benign_errors": sum(r["error"] is not None for r in ben),
                "utility": u, "utility_worst_case": u_w,
                "attacked_runs": len(att), "attacked_errors": sum(r["error"] is not None for r in att),
                "utility_under_attack": ua, "utility_under_attack_worst_case": ua_w,
                "targeted_asr": asr, "targeted_asr_worst_case": asr_w,
            }
        out[defence] = d
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:18081/v1")
    ap.add_argument("--model", default="qwen3.5-9b-q8_0", help="model id sent to the server (llama-server --alias)")
    ap.add_argument("--name", default="local-qwen3.5-9b-q8_0",
                    help="pipeline / log name; must contain 'local' so the attack can address the model")
    ap.add_argument("--benchmark-version", default="v1.2.2")
    ap.add_argument("--suites", nargs="*", default=["banking", "slack", "travel", "workspace"])
    ap.add_argument("--subset", default="small", help="small | full | path to a JSON {suite: {user_tasks, injection_tasks}}")
    ap.add_argument("--defences", nargs="*", default=["none", "tool_guard"],
                    help="none tool_filter repeat_user_prompt spotlighting_with_delimiting tool_guard")
    ap.add_argument("--guard-policy", default="allow_all")
    ap.add_argument("--guard-policy-kwargs", default="{}")
    ap.add_argument("--guard-mode", default="veto", choices=["veto", "abort"])
    ap.add_argument("--attack", default="important_instructions")
    ap.add_argument("--no-benign", action="store_true")
    ap.add_argument("--no-attack", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--parallel-tool-calls", default="true", choices=["server", "true", "false"],
                    help="ask llama-server for several calls per turn; with one call per turn, a task needing more "
                         "than 16 calls cannot finish (ToolsExecutionLoop max_iters=15; travel user_task_19 needs 18)")
    ap.add_argument("--timeout", type=float, default=900.0, help="seconds per HTTP request")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if "local" not in args.name:
        sys.exit("--name must contain 'local' (agentdojo.models.MODEL_NAMES key used by the attacks)")

    t_start = time.time()
    chosen = select(args.benchmark_version, args.subset, args.suites)
    ptc = None if args.parallel_tool_calls == "server" else args.parallel_tool_calls == "true"
    client = openai.OpenAI(base_url=args.base_url, api_key="sk-no-key-required", timeout=args.timeout, max_retries=0)
    llm = LlamaServerLLM(client, args.model, name=args.name, temperature=args.temperature,
                         max_tokens=args.max_tokens, seed=args.seed, parallel_tool_calls=ptc)
    policy = load_policy(args.guard_policy, json.loads(args.guard_policy_kwargs)) if "tool_guard" in args.defences else None
    pipelines = {}
    for defence in args.defences:
        pipelines[defence] = build_pipeline(llm, defence, policy=policy, guard_mode=args.guard_mode)

    plan = {"benign": 0, "attacked": 0}
    for s, sel in chosen.items():
        suite = get_suite(args.benchmark_version, s)
        for defence, (pipe, _) in pipelines.items():
            attack = load_attack(args.attack, suite, pipe)
            for ut_id in sel["user_tasks"]:
                attack.get_injection_candidates(suite.get_user_task_by_id(ut_id))  # raises if not injectable
            for it_id in sel["injection_tasks"]:
                suite.get_injection_task_by_id(it_id)
            plan["benign"] += 0 if args.no_benign else len(sel["user_tasks"])
            plan["attacked"] += 0 if args.no_attack else len(sel["user_tasks"]) * len(sel["injection_tasks"])
    names = {d: (p.name, get_model_name_from_pipeline(p)) for d, (p, _) in pipelines.items()}
    print(json.dumps({"version": args.benchmark_version, "subset": args.subset, "suites": chosen,
                      "pipelines (name, attack addresses model as)": names, "runs": plan}, indent=1))
    if args.dry_run:
        print("dry run: no request sent, nothing written")
        return

    assert args.out, "--out is required"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    base = args.base_url.rsplit("/v1", 1)[0]
    server = {"health": http_json(base + "/health"), "models": http_json(args.base_url + "/models")}
    import agentdojo
    from importlib.metadata import version as pkg_version
    meta = {
        "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_start)), "args": vars(args), "server": server,
        "agentdojo": pkg_version("agentdojo"), "agentdojo_file": agentdojo.__file__, "openai": pkg_version("openai"),
        "harness_sha256_16": {p.name: sha16(p) for p in sorted((ROOT / "harness").glob("*.py"))} | {
            "run_agentdojo_local.py": sha16(Path(__file__))},
        "plan": plan, "selection": chosen,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    rows_f = (out / "runs.jsonl").open("a")
    rows: list[dict] = []
    logdir = out / "traces"

    def record(row: dict) -> None:
        rows.append(row)
        rows_f.write(json.dumps(row) + "\n")
        rows_f.flush()
        print(f"{time.strftime('%H:%M:%S', time.gmtime())}Z {row['defence']:>12} {row['suite']:>9} {row['user_task']:>14} "
              f"{str(row['injection_task']):>17} u={row['utility']} s={row['security']} err={row['error'] is not None} "
              f"{row['seconds']}s", flush=True)

    with QuietLogger(str(logdir)) as qlog:
        for s, sel in chosen.items():
            suite = get_suite(args.benchmark_version, s)
            for defence, (pipe, guard) in pipelines.items():
                attack = load_attack(args.attack, suite, pipe)
                jobs = [] if args.no_benign else [(ut, None) for ut in sel["user_tasks"]]
                jobs += [] if args.no_attack else [(ut, it) for ut in sel["user_tasks"] for it in sel["injection_tasks"]]
                for ut_id, it_id in jobs:
                    user_task = suite.get_user_task_by_id(ut_id)
                    if guard is not None:
                        guard.context = {"suite": s, "user_task": ut_id, "injection_task": it_id,
                                         "attack": None if it_id is None else args.attack}
                    n0, p0, c0 = llm.n_requests, llm.usage["prompt_tokens"], llm.usage["completion_tokens"]
                    t0 = time.time()
                    err, u, sec = None, None, None
                    try:
                        if it_id is None:
                            u, sec = run_task_without_injection_tasks(suite, pipe, user_task, logdir, True,
                                                                      args.benchmark_version)
                        else:
                            ures, sres = run_task_with_injection_tasks(suite, pipe, user_task, attack, logdir, True,
                                                                       [it_id], args.benchmark_version)
                            u, sec = ures[(ut_id, it_id)], sres[(ut_id, it_id)]
                    except Exception as e:  # noqa: BLE001
                        err = f"{type(e).__name__}: {e}"[:2000]
                        (out / "errors.log").open("a").write(f"--- {defence} {s} {ut_id} {it_id}\n{traceback.format_exc()}\n")
                    record({"defence": defence, "pipeline": pipe.name, "suite": s, "user_task": ut_id,
                            "injection_task": it_id, "attack": None if it_id is None else args.attack,
                            "utility": u, "security": None if it_id is None else sec, "error": err,
                            "seconds": round(time.time() - t0, 2), "requests": llm.n_requests - n0,
                            "prompt_tokens": llm.usage["prompt_tokens"] - p0,
                            "completion_tokens": llm.usage["completion_tokens"] - c0})
                if guard is not None:
                    with (out / f"guard_decisions_{defence}.jsonl").open("a") as f:
                        for dcs in guard.decisions:
                            f.write(json.dumps(dcs) + "\n")
                    guard.decisions.clear()
        logger_errors = qlog.errors
    rows_f.close()
    summary = {"summary": summarise(rows), "logged_errors": logger_errors[:50],
               "end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "minutes": round((time.time() - t_start) / 60, 1),
               "requests": llm.n_requests, "usage": llm.usage}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
