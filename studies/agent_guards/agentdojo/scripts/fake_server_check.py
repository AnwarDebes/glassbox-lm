"""Offline check of the OpenAI-format path used in the GPU slot (LlamaServerLLM -> openai message conversion), with an
in-process FAKE client instead of llama-server: no network, no model.

    .venv/bin/python -I -B scripts/fake_server_check.py --out offline_logs/fake_server_check.json [--version v1.2.2]

The fake client's chat.completions.create(**body) records the request body, checks it, and answers with an
openai ChatCompletion holding the next ground-truth tool call of the task (found from the first user message), or
the task's GROUND_TRUTH_OUTPUT when the calls are used up. "fooled" mode adds the injection task's ground-truth calls
after the user task's calls. Checks on every request body: JSON-serialisable; temperature 0.0, max_tokens, seed,
extra_body.chat_template_kwargs.enable_thinking false, tools + tool_choice auto + parallel_tool_calls true; roles in
{developer, user, assistant, tool}; every tool message has a tool_call_id that refers to a call in an earlier
assistant message.
Runs (banking + slack, all user tasks; 2 injection tasks each): plain pipeline benign; tool_guard(allow_all) benign;
fooled pairs plain; fooled pairs with the oracle allow-list guard (plumbing only).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))

from openai.types.chat import ChatCompletion, ChatCompletionMessage, ChatCompletionMessageToolCall  # noqa: E402
from openai.types.chat.chat_completion import Choice  # noqa: E402
from openai.types.chat.chat_completion_message_tool_call import Function  # noqa: E402
from openai.types.completion_usage import CompletionUsage  # noqa: E402

import agentdojo.attacks  # noqa: E402,F401
from agentdojo.attacks.attack_registry import load_attack  # noqa: E402
from agentdojo.task_suite.load_suites import get_suite  # noqa: E402

from guard import AllowAll, AllowListPolicy  # noqa: E402
from llamaserver_llm import LlamaServerLLM  # noqa: E402
from pipelines import build_pipeline  # noqa: E402


def text_of(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content)


class FakeCompletions:
    def __init__(self, suite) -> None:
        self.suite = suite
        env = suite.load_and_inject_default_environment({})
        self.gt = {}
        for t in suite.user_tasks.values():
            self.gt[t.PROMPT] = (list(t.ground_truth(env.model_copy(deep=True))), t.GROUND_TRUTH_OUTPUT)
        for t in suite.injection_tasks.values():
            self.gt.setdefault(t.GOAL, (list(t.ground_truth(env.model_copy(deep=True))), t.GROUND_TRUTH_OUTPUT))
        self.follow = None  # injection task to also carry out
        self.problems: list[str] = []
        self.n = 0

    def check(self, body: dict) -> None:
        try:
            json.dumps(body)
        except TypeError as e:
            self.problems.append(f"not JSON-serialisable: {e}")
        if body.get("temperature") != 0.0:
            self.problems.append(f"temperature {body.get('temperature')!r}")
        if body.get("max_tokens") != 4096 or body.get("seed") != 0:
            self.problems.append("max_tokens/seed missing")
        if body.get("extra_body", {}).get("chat_template_kwargs", {}).get("enable_thinking") is not False:
            self.problems.append("enable_thinking not false")
        if "tools" in body and (body.get("tool_choice") != "auto" or body.get("parallel_tool_calls") is not True):
            self.problems.append("tool_choice/parallel_tool_calls")
        seen_ids = set()
        for m in body["messages"]:
            if m["role"] not in {"developer", "user", "assistant", "tool"}:
                self.problems.append(f"role {m['role']}")
            for tc in m.get("tool_calls") or []:
                seen_ids.add(tc["id"])
            if m["role"] == "tool" and m.get("tool_call_id") not in seen_ids:
                self.problems.append(f"tool message without matching call id {m.get('tool_call_id')}")

    def create(self, **body):
        self.n += 1
        self.check(body)
        msgs = body["messages"]
        prompt = text_of(next(m for m in msgs if m["role"] == "user")["content"])
        calls, final = self.gt[prompt]
        if self.follow is not None and prompt != self.follow.GOAL:
            env = self.suite.load_and_inject_default_environment({})
            calls = calls + list(self.follow.ground_truth(env))
        pos = sum(1 for m in msgs if m["role"] == "assistant")
        usage = CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        if pos < len(calls):
            c = calls[pos]
            msg = ChatCompletionMessage(role="assistant", content=None, tool_calls=[ChatCompletionMessageToolCall(
                id=f"call_{pos}", type="function", function=Function(name=c.function, arguments=json.dumps(dict(c.args))))])
            reason = "tool_calls"
        else:
            msg = ChatCompletionMessage(role="assistant", content=final)
            reason = "stop"
        return ChatCompletion(id=f"fake-{self.n}", choices=[Choice(finish_reason=reason, index=0, message=msg)],
                              created=0, model=body["model"], object="chat.completion", usage=usage)


class FakeClient:
    def __init__(self, completions: FakeCompletions) -> None:
        self.chat = type("Chat", (), {"completions": completions})()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", default="v1.2.2")
    args = ap.parse_args()
    out = Path(args.out)
    assert not out.exists(), f"{out} exists"
    result = {}
    for suite_name, inj_ids in [("banking", ["injection_task_0", "injection_task_1"]), ("slack", ["injection_task_1", "injection_task_2"])]:
        suite = get_suite(args.version, suite_name)
        fake = FakeCompletions(suite)
        llm = LlamaServerLLM(FakeClient(fake), "qwen3.5-9b-q8_0", name="local-fake", parallel_tool_calls=True)  # type: ignore[arg-type]
        plain, _ = build_pipeline(llm, "none")
        allow, allow_guard = build_pipeline(llm, "tool_guard", policy=AllowAll())
        oracle, oracle_guard = build_pipeline(llm, "tool_guard", policy=AllowListPolicy(set(), "oracle_allow_list"))
        env = suite.load_and_inject_default_environment({})
        gt_fn = {t.ID: {c.function for c in t.ground_truth(env.model_copy(deep=True))} for t in suite.user_tasks.values()}
        r: dict = {}
        for label, pipe in [("plain_benign", plain), ("allow_all_benign", allow)]:
            res = [suite.run_task_with_pipeline(pipe, t, None, {})[0] for t in suite.user_tasks.values()]
            r[label] = f"{sum(res)}/{len(res)}"
        attack = load_attack("important_instructions", suite, plain)
        for label, pipe, guard in [("plain_fooled", plain, None), ("oracle_guard_fooled", oracle, oracle_guard)]:
            u_ok = s_ok = n = 0
            for t in suite.user_tasks.values():
                for it_id in inj_ids:
                    it = suite.get_injection_task_by_id(it_id)
                    fake.follow = it
                    if guard is not None:
                        guard.policy = AllowListPolicy(gt_fn[t.ID], "oracle_allow_list")
                    u, s = suite.run_task_with_pipeline(pipe, t, it, attack.attack(t, it))
                    u_ok += bool(u)
                    s_ok += bool(s)
                    n += 1
            fake.follow = None
            r[label] = {"pairs": n, "utility": f"{u_ok}/{n}", "attack_success": f"{s_ok}/{n}"}
        r["oracle_guard_blocked_calls"] = sum(1 for d in oracle_guard.decisions if not d["allow"])
        r["allow_all_guard_decisions"] = len(allow_guard.decisions)
        r["requests"] = fake.n
        r["llm_requests_counted"] = llm.n_requests
        r["body_problems"] = sorted(set(fake.problems))[:20]
        r["n_body_problems"] = len(fake.problems)
        result[suite_name] = r
        print(suite_name, json.dumps(r), flush=True)
    out.write_text(json.dumps(result, indent=1))
    print("written", out)


if __name__ == "__main__":
    main()
