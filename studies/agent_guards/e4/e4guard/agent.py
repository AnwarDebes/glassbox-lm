"""The E4 guarded agent for tau2-bench v1.0.1 (taubench venv only).

GuardedAgent subclasses tau2's LLMAgent (same system prompt, same `generate` call, same message format), and adds:
  1. One call at a time. A model message with several tool calls is kept whole in the agent's private context
     (so the LLM sees exactly what it wrote), but the calls go to the environment one per message, in order. So
     each call is checked with the facts after the calls before it, as policy.md:20 asks. tau2 runs the calls of
     one message in the same order, so the executed sequence and the database outcome do not change.
  2. The guard. Every call is checked (models.decide) on trusted facts (facts.FactState). The facts are built from
     the orchestrator's trajectory only: the messages this agent emitted and the messages it received.
  3. Denied calls never reach the environment and never enter the trajectory. This matters: tau2's evaluator
     replays every mutating call of the trajectory (environment.py set_state), so a denied write left in the
     trajectory would be executed at scoring. The agent instead gets a synthetic tool message with the denial
     text in its private context, and the LLM is asked again (at most `max_retries` times per turn; then a fixed
     message to the user).
  4. Logging. Every guard verdict (with the live fact record's bit vector and sha256, and the monitor-only facts),
     every model call (latency, usage, sha256 of the request), every denied call and every construction of the agent
     (one per tau2 simulation attempt, so re-runs by tau2's run_with_retry are counted) goes to a JSONL log, and the
     emitted message's raw_data carries the verdict too.
A0 uses the same class in monitor mode, so all arms share one code path; A0's LLM requests are the ones stock
LLMAgent would send (no denial ever happens in A0).
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from datetime import datetime, timezone

from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.message import AssistantMessage, MultiToolMessage, ToolMessage, UserMessage
from tau2.utils.llm_utils import generate, to_litellm_messages

from . import models
from .facts import FactState

FALLBACK_TEXT = ("I'm sorry, I can't complete that action under our policy right now. "
                 "Is there anything else I can help you with?")
_LOG_LOCK = threading.Lock()


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _event_of_message(m) -> list[dict]:
    """Trajectory message -> facts events (same mapping as facts.events_from_message_dicts)."""
    from .facts import events_from_message_dicts

    if isinstance(m, MultiToolMessage):
        return events_from_message_dicts([t.model_dump() for t in m.tool_messages])
    return events_from_message_dicts([m.model_dump()])


class GuardedAgent(LLMAgent):
    def __init__(self, tools, domain_policy, llm, llm_args=None, *, arm="A0", learned=None, proposer=None,
                 log_path=None, task_id=None, max_retries=3):
        super().__init__(tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args)
        if arm not in models.ARMS:
            raise ValueError(arm)
        self.arm = arm
        self.learned = learned
        self.proposer = proposer          # None = the LLM (tau2's generate); else a callable(state) -> AssistantMessage
        self.log_path = log_path
        self.task_id = task_id
        self.max_retries = max_retries
        self.facts = FactState()
        self.queue = []                   # tool calls of the last model message still to send
        self.carry = None                 # (text, cost, usage, raw_data) of that model message, for its first call
        self.denied_since_emit = []
        self.stats = {"model_calls": 0, "denied": 0, "fallbacks": 0, "multi_call_model_messages": 0}
        self.verdicts = []                # in memory, for tests and summaries (the JSONL log is the record)
        self._log({"event": "agent_init", "learned": bool(learned), "max_retries": max_retries})

    # ------------------------------------------------------------------ logging
    def _log(self, rec: dict) -> None:
        if not self.log_path:
            return
        rec = {"utc": utc(), "task": self.task_id, "arm": self.arm, **rec}
        with _LOG_LOCK:
            with open(self.log_path, "a") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")

    # ------------------------------------------------------------------ state
    def get_init_state(self, message_history=None):
        state = super().get_init_state(message_history=message_history)
        for m in state.messages:
            for e in _event_of_message(m):
                self.facts.update(e)
        return state

    def _absorb(self, message, state) -> None:
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        elif message is not None:
            state.messages.append(message)
        if message is not None:
            for e in _event_of_message(message):
                self.facts.update(e)

    # ------------------------------------------------------------------ the model call
    def _propose(self, state) -> AssistantMessage:
        msgs = state.system_messages + state.messages
        req = {"model": self.llm, "messages": to_litellm_messages(msgs),
               "tools": [t.openai_schema for t in self.tools] if self.tools else None}
        h = hashlib.sha256(json.dumps(req, sort_keys=True, default=str).encode()).hexdigest()
        t0 = time.perf_counter()
        if self.proposer is None:
            m = generate(model=self.llm, tools=self.tools, messages=msgs, call_name="agent_response", **self.llm_args)
        else:
            m = self.proposer(state)
        self.stats["model_calls"] += 1
        self._log({"event": "model_call", "request_sha256": h, "ms": round(1000 * (time.perf_counter() - t0), 1),
                   "usage": getattr(m, "usage", None), "n_tool_calls": len(m.tool_calls or []),
                   "has_text": bool(m.content and m.content.strip())})
        return m

    # ------------------------------------------------------------------ one turn
    def generate_next_message(self, message, state):
        if isinstance(message, UserMessage) and message.is_audio:
            raise ValueError("audio is not supported")
        self._absorb(message, state)
        retries = 0
        while True:
            while self.queue:
                tc = self.queue.pop(0)
                f = self.facts.facts_for_call(tc.name, tc.arguments)
                v = models.decide(self.arm, f, self.learned)
                self.verdicts.append({"name": tc.name, "args": tc.arguments, "allow": v["allow"],
                                      "code_rules": v["code_rules"], "veto_rules": v["veto_rules"],
                                      "learned": v["learned"], "is_write": f["is_write"], "facts": f,
                                      "monitor": v["monitor"]})
                self._log({"event": "verdict", "call": {"id": tc.id, "name": tc.name, "args": tc.arguments},
                           "allow": v["allow"], "code_rules": v["code_rules"], "veto_rules": v["veto_rules"],
                           "veto_clauses": v["veto_clauses"], "learned": v["learned"], "monitor": v["monitor"],
                           "is_write": f["is_write"], "x": models.bits_string(f), "facts_sha256": models.facts_sha256(f)})
                if v["allow"]:
                    text, cost, usage, raw = self.carry if self.carry else (None, 0.0, None, None)
                    self.carry = None
                    raw = dict(raw or {})
                    raw["e4_guard"] = {"arm": self.arm, "verdict": {k: v[k] for k in ("allow", "code_rules", "veto_rules",
                                                                                     "veto_clauses", "learned")},
                                       "denied_before": self.denied_since_emit}
                    self.denied_since_emit = []
                    out = AssistantMessage(role="assistant", content=text, tool_calls=[tc], cost=cost or 0.0,
                                           usage=usage or {"completion_tokens": 0, "prompt_tokens": 0}, raw_data=raw)
                    for e in _event_of_message(out):
                        self.facts.update(e)
                    return out, state
                # denied: a synthetic result in the private context only; the call never runs
                self.stats["denied"] += 1
                self.denied_since_emit.append({"call": {"name": tc.name, "args": tc.arguments},
                                               "code_rules": v["code_rules"], "veto_rules": v["veto_rules"],
                                               "learned": v["learned"]})
                state.messages.append(ToolMessage(id=tc.id, role="tool", content=v["denial_text"], error=True,
                                                  requestor="assistant"))
                if self.carry and not self.queue:
                    self.carry = None
            # nothing queued: ask the model (again)
            if self.denied_since_emit:
                if retries >= self.max_retries:
                    self.stats["fallbacks"] += 1
                    out = AssistantMessage(role="assistant", content=FALLBACK_TEXT, cost=0.0,
                                           usage={"completion_tokens": 0, "prompt_tokens": 0},
                                           raw_data={"e4_guard": {"arm": self.arm, "fallback": True,
                                                                  "denied_before": self.denied_since_emit}})
                    self._log({"event": "fallback", "denied": len(self.denied_since_emit)})
                    self.denied_since_emit = []
                    state.messages.append(out)
                    for e in _event_of_message(out):
                        self.facts.update(e)
                    return out, state
                retries += 1
            m = self._propose(state)
            state.messages.append(m)                    # the private context keeps the model's message whole
            if not m.tool_calls:
                raw = dict(m.raw_data or {})
                if self.denied_since_emit:
                    raw["e4_guard"] = {"arm": self.arm, "denied_before": self.denied_since_emit}
                    self.denied_since_emit = []
                out = m.model_copy(update={"raw_data": raw or m.raw_data})
                for e in _event_of_message(out):
                    self.facts.update(e)
                return out, state
            self.queue = list(m.tool_calls)
            self.stats["multi_call_model_messages"] += int(len(m.tool_calls) > 1)
            self.carry = (m.content if (m.content and m.content.strip()) else None, m.cost, m.usage, m.raw_data)


def make_factory(arm: str, learned_path: str | None = None, log_path: str | None = None, max_retries: int = 3):
    """A tau2 agent factory: factory(tools, domain_policy, **kw) -> GuardedAgent (one per simulation)."""
    models.check_arm_model(arm, learned_path)
    learned = models.load_learned(learned_path, guarded_tm=True) if learned_path else None

    def factory(tools, domain_policy, **kw):
        task = kw.get("task")
        return GuardedAgent(tools=tools, domain_policy=domain_policy, llm=kw.get("llm"), llm_args=kw.get("llm_args"),
                            arm=arm, learned=learned, log_path=log_path, task_id=getattr(task, "id", None),
                            max_retries=max_retries)

    return factory
