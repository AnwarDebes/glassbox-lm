"""Paired replay (DESIGN 1.7): LlamaServerLLM with a store that maps sha256(request body) -> reply.

Every arm sends the same requests as the no-defence arm until its first veto; those replies come from the store,
later requests go to the server and are stored too. This pairs the arms and saves GPU time. A stored reply is used
only for a byte-identical canonical request body (json.dumps(sort_keys=True)), so a replayed reply is exactly what
the server answered to that request.

Logged per request (DESIGN 1.7, "Logged per call"): UTC time, context (stage, arm, suite, tasks), sha256 of the body,
store hit or miss, token usage, latency, the parsed tool calls, and the tool-call parse check used by E2-0:
- parsed: the reply carries at least one tool call parsed by the server;
- unparsed: no parsed tool call, but the text holds tool-call markup (Qwen3.5's <tool_call> / <function=...> or a JSON
  object with "name" and "arguments"): the server failed to parse a call the model tried to make;
- text: neither (a final answer).
rev2 (2026-10-21): the store itself lives in replay_store.py (no third-party imports, shared with the A3 optimizer and
the D6 rule writer); every exception of a request is recorded in `episode_errors` before it is raised, so an error that
upstream folds into (utility False, security True) is still visible to the error rule's no-error reading; the call log
also records the thinking check (reasoning_content, <think>), whether every tool-call argument string is a JSON object,
the number of messages and the body size; `bodies_seen` keeps distinct bodies with their context for E2-0's repeat
check (20 spread over the 4 suites, see run_e2_gpu.repeat_selection).
rev3 (2026-10-21): `pass_requests` counts the agent requests of each pass of an episode. AgentDojo's
TaskSuite.run_task_with_pipeline reruns the whole pipeline up to 3 times when a pass ends without a final text, on the
environment the earlier pass left; a pass starts with a request whose messages hold no assistant message yet, and one
pass sends at most 1 + 15 agent requests (ToolsExecutionLoop's max_iters). The driver clears it per episode, like
`episode_errors`; the turn cap is tested per pass (registration section 2).
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../../.."))

import json
import re
import sys
import time
from collections.abc import Sequence
from pathlib import Path

AGENTDOJO_HARNESS = Path(f"{_REPO_ROOT}/studies/agent_guards/agentdojo/harness")
if str(AGENTDOJO_HARNESS) not in sys.path:
    sys.path.insert(0, str(AGENTDOJO_HARNESS))

from agentdojo.agent_pipeline.llms.openai_llm import (
    _function_to_openai,
    _message_to_openai,
    _openai_to_assistant_message,
)
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import ChatMessage
from llamaserver_llm import (
    LlamaServerLLM,
    _request,
)
from openai.types.chat import ChatCompletion

from .replay_store import (
    ReplayStore,
    body_key,
    canonical,
)

MARKUP = re.compile(r"<tool_call>|<function=|\"name\"\s*:\s*\"[A-Za-z_]+\"\s*,\s*\"(arguments|parameters)\"")


def parse_class(completion: ChatCompletion) -> str:
    m = completion.choices[0].message
    if m.tool_calls:
        return "parsed"
    if m.content and MARKUP.search(m.content):
        return "unparsed"
    return "text"


def reasoning_of(message) -> str:
    """The server's separate thinking text, if any (llama-server puts it in `reasoning_content`)."""
    extra = getattr(message, "model_extra", None) or {}
    r = extra.get("reasoning_content") or getattr(message, "reasoning_content", None)
    return r if isinstance(r, str) else ""


def args_json_ok(message) -> bool | None:
    """True when every tool call's argument string parses as a JSON object; None when there is no tool call."""
    if not message.tool_calls:
        return None
    for t in message.tool_calls:
        try:
            if not isinstance(json.loads(t.function.arguments or ""), dict):
                return False
        except (json.JSONDecodeError, TypeError):
            return False
    return True


def log_fields(completion: ChatCompletion) -> dict:
    m = completion.choices[0].message
    content = m.content or ""
    return {"parse": parse_class(completion), "finish": completion.choices[0].finish_reason,
            "reasoning": bool(reasoning_of(m)), "think_tag": "<think>" in content or "</think>" in content,
            "args_json_ok": args_json_ok(m)}


class StoreLLM(LlamaServerLLM):
    def __init__(self, client, model: str, name: str, store: ReplayStore | None, call_log: Path | None = None,
                 **kw) -> None:
        super().__init__(client, model, name, **kw)
        self.store = store
        self.context: dict = {}
        self.hits = 0
        self.misses = 0
        self._log = call_log.open("a") if call_log is not None else None
        self.bodies_seen: list[dict] = []  # distinct bodies with their context, kept for the E2-0 repeat check
        self._bodies_keys: set[str] = set()
        self.keep_bodies = 0
        self.episode_errors: list[str] = []  # every request exception since the driver last cleared it
        self.pass_requests: list[int] = []  # agent requests per pass of the current episode (rev3)

    def build_body(self, messages: Sequence[ChatMessage], runtime: FunctionsRuntime) -> dict:
        openai_messages = [_message_to_openai(message, self.model) for message in messages]
        openai_tools = [_function_to_openai(tool) for tool in runtime.functions.values()]
        body: dict = {"model": self.model, "messages": openai_messages, "temperature": self.temperature,
                      "max_tokens": self.max_tokens,
                      "extra_body": {"chat_template_kwargs": {"enable_thinking": self.enable_thinking}}}
        if self.seed is not None:
            body["seed"] = self.seed
        if openai_tools:
            body["tools"] = openai_tools
            body["tool_choice"] = "auto"
            if self.parallel_tool_calls is not None:
                body["parallel_tool_calls"] = self.parallel_tool_calls
        return body

    def send(self, body: dict, use_store: bool = True) -> tuple[ChatCompletion, bool, float]:
        key = body_key(body)
        hit = self.store.get(key) if (use_store and self.store is not None) else None
        t0 = time.time()
        if hit is not None:
            return ChatCompletion.model_validate(hit["reply"]), True, 0.0
        try:
            completion = _request(self.client, body)
        except Exception as e:
            self.episode_errors.append(f"{type(e).__name__}: {e}"[:500])
            raise
        ms = (time.time() - t0) * 1000
        if use_store and self.store is not None:
            self.store.put(key, completion.model_dump(mode="json"),
                           {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "ms": round(ms),
                            "context": self.context})
        return completion, False, ms

    def query(self, query: str, runtime: FunctionsRuntime, env: Env = EmptyEnv(),  # noqa: B008
              messages: Sequence[ChatMessage] = [], extra_args: dict = {}):  # noqa: B006 (AgentDojo's element signature; never mutated)
        body = self.build_body(messages, runtime)
        if self.keep_bodies and len(self.bodies_seen) < self.keep_bodies:
            k = body_key(body)
            if k not in self._bodies_keys:
                self._bodies_keys.add(k)
                self.bodies_seen.append({"key": k, "context": dict(self.context), "n_messages": len(body["messages"]),
                                         "chars": len(canonical(body)), "body": json.loads(canonical(body))})
        if not any(m.get("role") == "assistant" for m in messages):  # the first request of a pass
            self.pass_requests.append(0)
        if self.pass_requests:
            self.pass_requests[-1] += 1
        completion, hit, ms = self.send(body)
        self.n_requests += 1
        if hit:
            self.hits += 1
        else:
            self.misses += 1
            if completion.usage is not None:
                self.usage["prompt_tokens"] += completion.usage.prompt_tokens or 0
                self.usage["completion_tokens"] += completion.usage.completion_tokens or 0
        if self._log is not None:
            m = completion.choices[0].message
            self._log.write(json.dumps({
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **self.context, "key": body_key(body),
                "hit": hit, "ms": round(ms), "usage": completion.usage.model_dump() if completion.usage else None,
                **log_fields(completion), "n_messages": len(body["messages"]), "body_chars": len(canonical(body)),
                "tool_calls": [{"name": t.function.name, "arguments": t.function.arguments[:500]} for t in (m.tool_calls or [])],
                "content": (m.content or "")[:500]}, default=str) + "\n")
            self._log.flush()
        output = _openai_to_assistant_message(completion.choices[0].message)
        return query, runtime, env, [*messages, output], extra_args


class _Completions:
    def __init__(self, outer: StoreClient) -> None:
        self.outer = outer

    def create(self, **kwargs):
        return self.outer.create(**kwargs)


class StoreClient:
    """A client wrapper for elements that call the client directly (upstream's OpenAILLMToolFilter, arm D1): the same
    store and call log as StoreLLM, so D1's filter call is replayed, logged and counted like every other call."""

    def __init__(self, client, store: ReplayStore | None, llm: StoreLLM, role: str = "tool_filter") -> None:
        self.client, self.store, self.llm, self.role = client, store, llm, role
        self.chat = type("Chat", (), {"completions": _Completions(self)})()

    def create(self, **kwargs):
        key = body_key(kwargs)
        hit = self.store.get(key) if self.store is not None else None
        t0 = time.time()
        if hit is not None:
            completion = ChatCompletion.model_validate(hit["reply"])
        else:
            try:
                completion = self.client.chat.completions.create(**kwargs)
            except Exception as e:
                self.llm.episode_errors.append(f"{self.role}: {type(e).__name__}: {e}"[:500])
                raise
            if self.store is not None:
                self.store.put(key, completion.model_dump(mode="json"),
                               {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                "ms": round((time.time() - t0) * 1000), "context": {**self.llm.context, "role": self.role}})
        ms = 0.0 if hit is not None else (time.time() - t0) * 1000
        self.llm.n_requests += 1
        if hit is not None:
            self.llm.hits += 1
        else:
            self.llm.misses += 1
            if completion.usage is not None:
                self.llm.usage["prompt_tokens"] += completion.usage.prompt_tokens or 0
                self.llm.usage["completion_tokens"] += completion.usage.completion_tokens or 0
        if self.llm._log is not None:
            m = completion.choices[0].message
            self.llm._log.write(json.dumps({
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **self.llm.context, "role": self.role,
                "key": key, "hit": hit is not None, "ms": round(ms),
                "usage": completion.usage.model_dump() if completion.usage else None, **log_fields(completion),
                "n_messages": len(kwargs.get("messages") or []), "body_chars": len(canonical(kwargs)),
                "tool_calls": [], "content": (m.content or "")[:2000]},
                default=str) + "\n")
            self.llm._log.flush()
        return completion
