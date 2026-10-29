"""An in-process FAKE OpenAI client (no network, no model), so arms that need the LLM element (D1 tool_filter, the
paired-replay store, the GPU driver's code paths) can be run offline through AgentDojo's real pipeline. A stand-in:
its numbers test plumbing only.

- An agent request (tool_choice 'auto') is answered with the next planned ground-truth call of the task named by the
  first user message (user task ground truth, then the injection task's when `follow` is set), one call per turn; the
  plan position is the number of earlier assistant messages that carry tool calls (so the tool filter's own reply does
  not shift it). When the plan is used up, the task's ground-truth output is the answer.
- A tool-filter request (tool_choice 'none') is answered with every tool name it was offered (the weakest filter),
  or with the user task's ground-truth tool names when `filter_mode='oracle'` (a plumbing test that uses benchmark
  answers; never a defence).
"""

from __future__ import annotations

import json

from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import Function
from openai.types.completion_usage import CompletionUsage


def _text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content)


class FakeCompletions:
    def __init__(self, suite, filter_mode: str = "all") -> None:
        self.suite = suite
        env = suite.load_and_inject_default_environment({})
        self.gt = {}
        for t in suite.user_tasks.values():
            self.gt[t.PROMPT] = (list(t.ground_truth(env.model_copy(deep=True))), t.GROUND_TRUTH_OUTPUT)
        self.follow = None
        self.filter_mode = filter_mode
        self.n = 0
        self.n_filter = 0
        self.bodies: list[dict] = []
        self.keep_bodies = False

    def create(self, **body):
        self.n += 1
        if self.keep_bodies:
            self.bodies.append(json.loads(json.dumps(body, default=str)))
        msgs = body["messages"]
        prompt = _text(next(m for m in msgs if m["role"] == "user")["content"])
        calls, final = self.gt[prompt]
        usage = CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        if body.get("tool_choice") == "none":
            self.n_filter += 1
            names = [t["function"]["name"] for t in body.get("tools") or []]
            if self.filter_mode == "oracle":
                names = sorted({c.function for c in calls})
            msg = ChatCompletionMessage(role="assistant", content=", ".join(names))
            return ChatCompletion(id=f"fake-{self.n}", choices=[Choice(finish_reason="stop", index=0, message=msg)],
                                  created=0, model=body["model"], object="chat.completion", usage=usage)
        if self.follow is not None:
            env = self.suite.load_and_inject_default_environment({})
            calls = calls + list(self.follow.ground_truth(env))
        pos = sum(1 for m in msgs if m["role"] == "assistant" and m.get("tool_calls"))
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
