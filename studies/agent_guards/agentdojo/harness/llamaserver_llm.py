"""OpenAILLM variant for a local llama.cpp llama-server (OpenAI-compatible, native tool calls parsed by the server).

Why not plain OpenAILLM (agentdojo 0.1.35, agent_pipeline/llms/openai_llm.py):
  * chat_completion_request sends `temperature=temperature or NOT_GIVEN` (line 162): 0.0 is falsy, so NO temperature
    is sent and llama-server samples at its default 0.80 (llama.cpp common/common.h:239). Fixed upstream only on main
    (commit 462c88d, PR #141, 2026-02-03), not released. Here temperature is always sent.
  * No max_tokens is sent; llama-server then generates until EOS or a full context. Here max_tokens is capped.
  * Thinking is switched off per request (chat_template_kwargs enable_thinking false), in addition to the server's
    --reasoning-budget 0.
Everything else (message conversion, tool schemas, parsing of the reply) reuses the upstream helper functions, so the
conversation the model sees is the same as with AgentDojo's own OpenAILLM / "vllm_parsed" provider.
Role "developer" (used by upstream for the system prompt, openai_llm.py:79-82) is mapped to "system" by llama-server
itself for all templates except GPT-OSS (llama.cpp b11742 common/chat.cpp:1347-1350), so it is not changed here.
"""

from __future__ import annotations

from collections.abc import Sequence

import openai
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_random_exponential

from agentdojo.agent_pipeline.llms.openai_llm import (
    OpenAILLM,
    _function_to_openai,
    _message_to_openai,
    _openai_to_assistant_message,
)
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import ChatMessage


@retry(
    wait=wait_random_exponential(multiplier=1, max=40),
    stop=stop_after_attempt(3),
    reraise=True,
    retry=retry_if_not_exception_type((openai.BadRequestError, openai.UnprocessableEntityError)),
)
def _request(client: openai.OpenAI, body: dict):
    return client.chat.completions.create(**body)


class LlamaServerLLM(OpenAILLM):
    def __init__(
        self,
        client: openai.OpenAI,
        model: str,
        name: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        seed: int | None = 0,
        enable_thinking: bool = False,
        parallel_tool_calls: bool | None = None,
    ) -> None:
        super().__init__(client, model, temperature=temperature)
        self.name = name  # pipeline / log-folder name; must contain a key of agentdojo.models.MODEL_NAMES ("local")
        self.max_tokens = max_tokens
        self.seed = seed
        self.enable_thinking = enable_thinking
        self.parallel_tool_calls = parallel_tool_calls
        self.n_requests = 0
        self.usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        openai_messages = [_message_to_openai(message, self.model) for message in messages]
        openai_tools = [_function_to_openai(tool) for tool in runtime.functions.values()]
        body: dict = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": self.enable_thinking}},
        }
        if self.seed is not None:
            body["seed"] = self.seed
        if openai_tools:
            body["tools"] = openai_tools
            body["tool_choice"] = "auto"
            if self.parallel_tool_calls is not None:
                body["parallel_tool_calls"] = self.parallel_tool_calls
        completion = _request(self.client, body)
        self.n_requests += 1
        if completion.usage is not None:
            self.usage["prompt_tokens"] += completion.usage.prompt_tokens or 0
            self.usage["completion_tokens"] += completion.usage.completion_tokens or 0
        output = _openai_to_assistant_message(completion.choices[0].message)
        return query, runtime, env, [*messages, output], extra_args
