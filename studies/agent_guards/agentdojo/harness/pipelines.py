"""Build AgentDojo 0.1.35 pipelines around a given model element, for the defences compared later.

  none, tool_filter, repeat_user_prompt, spotlighting_with_delimiting
      -> AgentPipeline.from_config(...) exactly as upstream (agent_pipeline/agent_pipeline.py:182-265), with our model
         object as `llm`. tool_filter requires an OpenAILLM subclass (line 209); LlamaServerLLM is one.
  transformers_pi_detector
      -> not available here (needs transformers + torch + a model download; not installed).
  tool_guard  (with a policy object)
      -> [SystemMessage, InitQuery, llm, ToolsExecutionLoop([GuardedToolsExecutor(policy, mode), llm])]
         i.e. the "none" pipeline with ToolsExecutor replaced by the guarded executor.
The pipeline name (used for log folders and by the attacks to address the model) is llm.name for "none" and
f"{llm.name}-{defence}" otherwise, as upstream does.
"""

from __future__ import annotations

from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, PipelineConfig, load_system_message
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.basic_elements import InitQuery, SystemMessage
from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop

from guard import GuardedToolsExecutor, ToolCallPolicy

UPSTREAM_DEFENCES = ("none", "tool_filter", "repeat_user_prompt", "spotlighting_with_delimiting")


def build_pipeline(
    llm: BasePipelineElement,
    defence: str,
    policy: ToolCallPolicy | None = None,
    guard_mode: str = "veto",
    system_message: str | None = None,
) -> tuple[AgentPipeline, GuardedToolsExecutor | None]:
    if llm.name is None:
        raise ValueError("the model element needs a name")
    if defence in UPSTREAM_DEFENCES:
        config = PipelineConfig(
            llm=llm,
            model_id=None,
            defense=None if defence == "none" else defence,
            system_message_name=None,
            system_message=system_message,
        )
        return AgentPipeline.from_config(config), None
    if defence == "transformers_pi_detector":
        raise ValueError("transformers_pi_detector needs transformers + torch + a model download (not installed)")
    if defence == "tool_guard":
        if policy is None:
            raise ValueError("tool_guard needs a policy")
        guard = GuardedToolsExecutor(policy, mode=guard_mode)  # type: ignore[arg-type]
        sys_msg = system_message if system_message is not None else load_system_message(None)
        pipeline = AgentPipeline(
            [SystemMessage(sys_msg), InitQuery(), llm, ToolsExecutionLoop([guard, llm])]
        )
        pipeline.name = f"{llm.name}-toolguard-{policy.name}-{guard_mode}"
        return pipeline, guard
    raise ValueError(f"unknown defence {defence!r}")
