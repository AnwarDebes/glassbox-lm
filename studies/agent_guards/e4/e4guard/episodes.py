"""tau2-side helpers for offline episodes (no LLM): direct gold trajectories and SimulationRun wrappers.

Only imported in the taubench venv (tau2 v1.0.1). Every tau2 import is inside a function, so the module can be
listed without tau2 installed.
"""

from __future__ import annotations

from datetime import datetime, timezone


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def gold_trajectory_direct(task, env_ctor, with_final: bool = True):
    """Execute the task's gold actions in a fresh environment and return (trajectory, stats).

    Same construction as taubench/scripts/smoke_offline.py:build_episode (greeting, each gold action as an
    assistant (or user) tool call followed by its tool message, one final agent message, the user's ###STOP###).
    No confirmation turns: this is the reward check, not the guard check (see scripted.py for that).
    """
    from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage
    from tau2.orchestrator.orchestrator import DEFAULT_FIRST_AGENT_MESSAGE

    env = env_ctor()
    ist = task.initial_state
    env.set_state(initialization_data=ist.initialization_data if ist is not None else None,
                  initialization_actions=ist.initialization_actions if ist is not None else None,
                  message_history=list(ist.message_history) if (ist is not None and ist.message_history) else [])
    actions = list(task.evaluation_criteria.actions or []) if task.evaluation_criteria else []
    traj = [DEFAULT_FIRST_AGENT_MESSAGE.model_copy(deep=True)]
    n_err = 0
    for i, a in enumerate(actions):
        tc = ToolCall(id=f"gold_{i}", name=a.name, arguments=dict(a.arguments), requestor=a.requestor)
        msg = (UserMessage(role="user", content=None, tool_calls=[tc]) if a.requestor == "user"
               else AssistantMessage(role="assistant", content=None, tool_calls=[tc]))
        resp = env.get_response(tc)
        n_err += int(bool(resp.error))
        traj.append(msg)
        traj.append(resp)
    if with_final:
        info = list(task.evaluation_criteria.communicate_info or []) if task.evaluation_criteria else []
        traj.append(AssistantMessage(role="assistant", content=("Done. " + " ".join(info)) if info else "Done."))
    traj.append(UserMessage(role="user", content="Thanks. ###STOP###"))
    return traj, {"n_actions": len(actions), "n_tool_errors": n_err}


def sim_run(task, traj, termination: str = "user_stop", sim_id: str | None = None):
    from tau2.data_model.simulation import SimulationRun, TerminationReason

    t = now_iso()
    return SimulationRun(id=sim_id or f"e4_{task.id}", task_id=task.id, timestamp=t, start_time=t, end_time=t,
                         duration=0.0, termination_reason=TerminationReason(termination), messages=traj)


def quiet_logs(level: str = "WARNING") -> None:
    """tau2 logs through loguru at DEBUG by default outside its CLI; keep offline runs readable."""
    import sys

    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level=level)
