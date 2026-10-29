"""Rows from AgentDojo run traces (TraceLogger JSON files): the facts of every executed tool call, as the guard would
have seen them, and the label of labels.agent_call_label (DESIGN E2, "Training rows", second source: the agent's own
D0 calls). Used on Qwen's D0 traces in the GPU stage; offline only on STAND-IN traces (scripted replays), whose rows
test the code and are never a finding.

A trace holds every message. For the k-th assistant message, the guard's view is the messages up to and including it
(earlier tool results only); every call in it that got a tool result (it was executed) gives one row. Calls of
unknown tools and calls that never ran (the last pass after ToolsExecutionLoop's max_iters) give no row.
"""

from __future__ import annotations

import json
from pathlib import Path

from .bits import fact_bits_on
from .facts import FactExtractor, call_function, call_id
from .labels import agent_call_label


def trace_files(root: Path) -> list[Path]:
    return sorted(p for p in Path(root).rglob("*.json") if p.is_file())


def rows_from_trace(trace: dict, suite, extractor: FactExtractor, source: str) -> tuple[list[dict], dict]:
    stats = {"calls": 0, "rows": 0, "unknown_tool": 0, "not_executed": 0}
    ut_id = trace["user_task_id"]
    if ut_id not in suite.user_tasks:
        return [], {**stats, "skipped": "not a user task"}
    ut = suite.user_tasks[ut_id]
    it_id = trace.get("injection_task_id")
    # ground truth on the clean environment and on the run's own injected one (the trace records the injections)
    env0 = suite.load_and_inject_default_environment({})
    env = suite.load_and_inject_default_environment(trace.get("injections") or {})
    user_gt = list(ut.ground_truth(env0.model_copy(deep=True)))
    user_gt_run = list(ut.ground_truth(env.model_copy(deep=True)))
    inj = None
    if it_id is not None and trace.get("attack_type") not in (None, "none"):
        it = suite.injection_tasks[it_id]
        inj = {"goal": it.GOAL, "gt": list(it.ground_truth(env.model_copy(deep=True)))}
    query = ut.PROMPT
    msgs = trace["messages"]
    executed = {m["tool_call_id"] for m in msgs if m.get("role") == "tool" and m.get("tool_call_id") is not None}
    rows = []
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant" or not m.get("tool_calls"):
            continue
        view = msgs[: i + 1]
        for c in m["tool_calls"]:
            stats["calls"] += 1
            if call_id(c) not in executed:
                stats["not_executed"] += 1
                continue
            f = extractor.extract(query, c, view)
            if f.kind is None:
                stats["unknown_tool"] += 1
                continue
            label, reason = agent_call_label(c, extractor.table, inj, query, user_gt, user_gt_run)
            rows.append({"source": source, "suite": trace["suite_name"], "user_task": ut_id,
                         "injection_task": it_id if inj else None, "attack": trace.get("attack_type") if inj else None,
                         "pipeline": trace.get("pipeline_name"), "step": len(rows), "function": call_function(c),
                         "args": c.get("args"), "label": label, "label_reason": reason,
                         "facts": f.to_json(), "facts_on": sorted(fact_bits_on(f)), "query_sha256": f.query_sha256,
                         "run_utility": trace.get("utility"), "run_security": trace.get("security")})
            stats["rows"] += 1
    return rows, stats


def load_trace(p: Path) -> dict:
    return json.loads(Path(p).read_text())
