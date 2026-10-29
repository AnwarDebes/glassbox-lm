"""Scripted stand-ins for the agent (no LLM), for training rows from the ground truth and for the offline end-to-end
tests of every arm. Not a model: numbers from them test the harness only.

- PlanReplay: replays the user task's ground-truth calls, one call per turn, then (if `follow` is set) the injection
  task's ground-truth calls ("always fooled"), then answers with the task's ground-truth output. Like
  harness_build/agentdojo/harness/replay.py, but it records where the user part ends (call ids 'u<k>' and 'i<k>'),
  and a planned call that the guard removed is simply not retried (the next turn moves on to the next planned call).
"""

from __future__ import annotations

import copy
from collections.abc import Sequence

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionCall, FunctionsRuntime
from agentdojo.types import (
    ChatAssistantMessage,
    ChatMessage,
    text_content_block_from_string,
)


class PlanReplay(BasePipelineElement):
    def __init__(self, suite, name: str = "local-replay") -> None:
        self.name = name
        self.by_prompt = {t.PROMPT: t for t in suite.user_tasks.values()}
        for t in suite.injection_tasks.values():
            self.by_prompt.setdefault(t.GOAL, t)
        self.follow = None
        self.last_plan_info: dict = {}

    def query(self, query: str, runtime: FunctionsRuntime, env: Env = EmptyEnv(),  # noqa: B008
              messages: Sequence[ChatMessage] = [], extra_args: dict = {}):  # noqa: B006 (AgentDojo's element signature; never mutated)
        if "plan" not in extra_args:
            task = self.by_prompt[query]
            user_plan = list(task.ground_truth(env))
            inj_plan = list(self.follow.ground_truth(env)) if self.follow is not None else []
            plan = [("u", k, c) for k, c in enumerate(user_plan)] + [("i", k, c) for k, c in enumerate(inj_plan)]
            extra_args = {**extra_args, "plan": plan, "pos": 0, "final": task.GROUND_TRUTH_OUTPUT}
            self.last_plan_info = {"n_user": len(user_plan), "n_injection": len(inj_plan),
                                   "user_plan": [(c.function, dict(c.args)) for c in user_plan],
                                   "injection_plan": [(c.function, dict(c.args)) for c in inj_plan]}
        pos, plan = extra_args["pos"], extra_args["plan"]
        if pos < len(plan):
            part, k, gt = plan[pos]
            call = FunctionCall(function=gt.function, args=copy.deepcopy(dict(gt.args)), id=f"{part}{k}")
            out = ChatAssistantMessage(role="assistant", content=[text_content_block_from_string("")], tool_calls=[call])
            extra_args = {**extra_args, "pos": pos + 1}
        else:
            out = ChatAssistantMessage(role="assistant", content=[text_content_block_from_string(extra_args["final"])],
                                       tool_calls=None)
        return query, runtime, env, [*messages, out], extra_args
