"""Tool-call guard for AgentDojo 0.1.35: looks at every tool call the model proposes and can veto it before it runs.

Where it plugs in (AgentDojo 0.1.35, see NOTES_guard_hook.md):
    pipeline = AgentPipeline([SystemMessage(sys_msg), InitQuery(), llm,
                              ToolsExecutionLoop([GuardedToolsExecutor(policy), llm])])
GuardedToolsExecutor takes the place of ToolsExecutor (agent_pipeline/tool_execution.py:46-115). The first model turn
runs before the loop, so its calls are handled by the first loop pass: the guard sees every proposed call before it runs.

Veto semantics ("veto" mode, the default):
  * A vetoed call is NOT run and is REMOVED from the assistant message. AgentDojo builds the function trace that some
    checks use (slack injection_task_5's security_from_traces, slack user_task_11's and the combined tasks'
    utility_from_traces) from the tool calls in the assistant messages, whether they ran or not
    (task_suite/task_suite.py:59-67). A vetoed call left in the message would therefore still count as made.
  * Allowed calls run exactly as in ToolsExecutor.
  * After the tool results, one user-role message lists the blocked calls and the reasons, so the model can carry on.
    (A tool-role message cannot be used: its tool_call_id would refer to a call that is no longer in the message.)
"abort" mode instead raises AbortAgentError at the first vetoed call (agent_pipeline/errors.py). The run stops, and
AgentDojo scores the environment as it is at that point (task_suite/task_suite.py:385-390).

Every verdict is appended to `decisions`, a list of dicts the caller can write out as training or audit data.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from agentdojo.agent_pipeline.errors import AbortAgentError
from agentdojo.agent_pipeline.tool_execution import ToolsExecutor, tool_result_to_str
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionCall, FunctionsRuntime
from agentdojo.types import (
    ChatAssistantMessage,
    ChatMessage,
    ChatUserMessage,
    text_content_block_from_string,
)


@dataclass
class Verdict:
    allow: bool
    reason: str = ""
    score: float | None = None


class ToolCallPolicy(Protocol):
    """A policy decides on one proposed call. It gets only what a deployed guard could see: the trusted user query,
    the proposed call, the conversation so far (incl. untrusted tool outputs) and the tool schemas (runtime.functions).
    It does NOT get the environment (that would be privileged state)."""

    name: str

    def decide(
        self, query: str, call: FunctionCall, messages: Sequence[ChatMessage], runtime: FunctionsRuntime
    ) -> Verdict: ...


class AllowAll:
    """Pass-through policy: the guarded pipeline must then behave exactly like the plain one."""

    name = "allow_all"

    def decide(self, query, call, messages, runtime) -> Verdict:
        return Verdict(True, "allow_all")


class AllowListPolicy:
    """Allows a call only if its function name is in a fixed set. With the set taken from a task's ground truth this is
    an ORACLE (it uses benchmark answers) and only serves to test the plumbing; it is not a deployable defence."""

    def __init__(self, allowed: set[str], name: str = "allow_list") -> None:
        self.allowed = set(allowed)
        self.name = name

    def decide(self, query, call, messages, runtime) -> Verdict:
        if call.function in self.allowed:
            return Verdict(True, "in allow-list")
        return Verdict(False, f"function {call.function!r} is not in the allow-list")


def iter_calls(call: FunctionCall):
    """The call and every FunctionCall nested in its arguments (FunctionsRuntime runs nested calls first,
    functions_runtime.py:54 and :279)."""
    yield call
    for value in call.args.values():
        if isinstance(value, FunctionCall):
            yield from iter_calls(value)


class GuardedToolsExecutor(ToolsExecutor):
    def __init__(
        self,
        policy: ToolCallPolicy,
        mode: Literal["veto", "abort"] = "veto",
        tool_output_formatter=tool_result_to_str,
    ) -> None:
        super().__init__(tool_output_formatter)
        self.policy = policy
        self.mode = mode
        self.decisions: list[dict] = []
        self.context: dict = {}  # set by the driver before each run (suite, user task, injection task, attack)

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        if len(messages) == 0 or messages[-1]["role"] != "assistant" or not messages[-1]["tool_calls"]:
            return super().query(query, runtime, env, messages, extra_args)
        last = messages[-1]
        turn = sum(1 for m in messages if m["role"] == "assistant")
        allowed: list[FunctionCall] = []
        blocked: list[tuple[FunctionCall, Verdict]] = []
        for call in last["tool_calls"]:
            verdicts = [self.policy.decide(query, c, messages, runtime) for c in iter_calls(call)]
            verdict = next((v for v in verdicts if not v.allow), verdicts[0])
            self.decisions.append(
                {
                    **self.context,
                    "policy": getattr(self.policy, "name", type(self.policy).__name__),
                    "turn": turn,
                    "call": call.model_dump(mode="json"),
                    "allow": verdict.allow,
                    "reason": verdict.reason,
                    "score": verdict.score,
                }
            )
            if verdict.allow:
                allowed.append(call)
            else:
                blocked.append((call, verdict))
        if not blocked:
            return super().query(query, runtime, env, messages, extra_args)
        if self.mode == "abort":
            call, verdict = blocked[0]
            raise AbortAgentError(
                f"Tool guard stopped the agent before {call.function}: {verdict.reason}", list(messages), env
            )
        content = last["content"] if last["content"] is not None else [text_content_block_from_string("")]
        rewritten = ChatAssistantMessage(role="assistant", content=content, tool_calls=allowed)
        query, runtime, env, messages, extra_args = super().query(
            query, runtime, env, [*messages[:-1], rewritten], extra_args
        )
        lines = []
        for call, verdict in blocked:
            args = ", ".join(f"{key}={value!r}" for key, value in call.args.items())
            lines.append(f"- {call.function}({args}): {verdict.reason}")
        note = (
            "[tool guard] These tool calls were blocked by the security policy and were NOT executed:\n"
            + "\n".join(lines)
            + "\nContinue with the user's original task without them."
        )
        feedback = ChatUserMessage(role="user", content=[text_content_block_from_string(note)])
        return query, runtime, env, [*messages, feedback], extra_args
