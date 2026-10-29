"""Offline stand-ins for the model, used only to test the AgentDojo plumbing without any LLM.

ReplayLLM replays a task's ground-truth tool calls (task.ground_truth(env), computed at the first model turn, as
GroundTruthPipeline does) one call per turn through the real AgentPipeline / ToolsExecutionLoop / ToolsExecutor path,
then answers with the task's GROUND_TRUTH_OUTPUT. It ignores injected text, so it acts as a "never fooled" agent.
With `follow_injection` set to an injection task it also replays that task's ground-truth calls after the user task's
calls: a scripted "always fooled" agent. Neither is a model; numbers from them only test the harness.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.base_tasks import BaseInjectionTask
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionCall, FunctionsRuntime
from agentdojo.task_suite.task_suite import TaskSuite
from agentdojo.types import ChatAssistantMessage, ChatMessage, text_content_block_from_string


class ReplayLLM(BasePipelineElement):
    def __init__(self, suite: TaskSuite, name: str = "local-replay") -> None:
        self.name = name
        self.by_prompt: dict[str, object] = {}
        for task in suite.user_tasks.values():
            self.by_prompt[task.PROMPT] = task
        for task in suite.injection_tasks.values():
            self.by_prompt.setdefault(task.GOAL, task)
        self.follow_injection: BaseInjectionTask | None = None

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        if "replay_plan" not in extra_args:
            task = self.by_prompt[query]
            plan = list(task.ground_truth(env))
            if self.follow_injection is not None:
                plan += list(self.follow_injection.ground_truth(env))
            extra_args = {**extra_args, "replay_plan": plan, "replay_pos": 0, "replay_final": task.GROUND_TRUTH_OUTPUT}
        pos, plan = extra_args["replay_pos"], extra_args["replay_plan"]
        if pos < len(plan):
            gt = plan[pos]
            call = FunctionCall(function=gt.function, args=copy.deepcopy(dict(gt.args)), id=f"replay_{pos}")
            out = ChatAssistantMessage(role="assistant", content=[text_content_block_from_string("")], tool_calls=[call])
            extra_args = {**extra_args, "replay_pos": pos + 1}
        else:
            out = ChatAssistantMessage(
                role="assistant", content=[text_content_block_from_string(extra_args["replay_final"])], tool_calls=None
            )
        return query, runtime, env, [*messages, out], extra_args
