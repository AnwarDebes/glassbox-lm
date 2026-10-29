"""Reading E4 run folders (pure Python; no tau2): results.json simulations, re-runs, end classes, guard logs.

Layout of one campaign (fixed in REGISTRATION_E4.md revision 2, section 12):
  runs/<campaign>/replay.sqlite                     the paired replay store (one per campaign)
  runs/<campaign>/<arm dir>/results.json            tau2's results (one trial per task)
  runs/<campaign>/<arm dir>/guard_log.jsonl         every verdict, model call, fallback and agent construction
  runs/<campaign>/<arm dir>_rerun/results.json      the one re-run of an arm's infrastructure errors and timeouts
                                                    (only when the arm has more than 3 of them; summary rule 9)
A re-run's simulations REPLACE the arm's simulations of the same tasks; every merged task is listed.

End classes (review issue 9): "infra" = infrastructure_error, unexpected_error or context_window_exceeded (tau2 gives
up after its retries; a context overflow or an unparsable tool call ends this way); "timeout" = the 1800 s limit;
"normal" = every other termination (user or agent stop, max_steps, too_many_errors, agent_error, user_error). Every
end scores what tau2's ENV evaluation gives; an infra or timeout end has no reward and scores 0.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

INFRA = {"infrastructure_error", "unexpected_error", "context_window_exceeded"}
TIMEOUT = {"timeout"}


def termination(sim: dict) -> str:
    t = sim.get("termination_reason")
    return str(t.get("value") if isinstance(t, dict) else t)


def end_class(sim: dict) -> str:
    t = termination(sim)
    return "infra" if t in INFRA else ("timeout" if t in TIMEOUT else "normal")


def db_reward(sim: dict) -> float:
    ri = sim.get("reward_info") or {}
    return float(ri.get("reward") or 0.0)


def _sims(path: Path) -> dict:
    d = json.loads(path.read_text())
    return {str(s["task_id"]): s for s in d["simulations"]}, d


def load_arm(arm_dir: Path) -> dict | None:
    """{"sims": {task: sim}, "tasks": {task: task dict}, "rerun_tasks": [...]} or None when there is no results.json."""
    p = Path(arm_dir) / "results.json"
    if not p.exists():
        return None
    sims, d = _sims(p)
    tasks = {str(t["id"]): t for t in d.get("tasks", [])}
    rr = Path(str(arm_dir) + "_rerun") / "results.json"
    rerun = []
    if rr.exists():
        rs, rd = _sims(rr)
        for t, s in rs.items():
            sims[t] = s
            rerun.append(t)
        tasks.update({str(t["id"]): t for t in rd.get("tasks", [])})
    return {"sims": sims, "tasks": tasks, "rerun_tasks": sorted(rerun, key=lambda z: int(z) if z.isdigit() else z)}


def canonical_messages(sim: dict) -> list:
    """The trajectory without timestamps, costs, usage or raw data: what the environment and the user saw."""
    out = []
    for m in sim.get("messages") or []:
        role = m.get("role")
        c = {"role": role, "content": m.get("content")}
        if m.get("tool_calls"):
            c["tool_calls"] = [{"id": t.get("id"), "name": t.get("name"),
                                "arguments": t.get("arguments")} for t in m["tool_calls"]]
        if role == "tool":
            c["id"], c["error"] = m.get("id"), bool(m.get("error"))
        out.append(c)
    return out


def trajectory_sha256(sim: dict) -> str:
    return hashlib.sha256(json.dumps(canonical_messages(sim), sort_keys=True, default=str).encode()).hexdigest()


def guard_events(arm_dir: Path) -> dict:
    """{task: [events]} from the arm's guard_log.jsonl (and its re-run's, which replaces the task's events)."""
    out = {}
    for d, replace in ((Path(arm_dir), False), (Path(str(arm_dir) + "_rerun"), True)):
        p = d / "guard_log.jsonl"
        if not p.exists():
            continue
        cur = {}
        for line in p.read_text().splitlines():
            e = json.loads(line)
            cur.setdefault(str(e.get("task")), []).append(e)
        if replace:
            out.update(cur)
        else:
            out = cur
    return out


def end_counts(arm: dict) -> dict:
    c = Counter(end_class(s) for s in arm["sims"].values())
    return {"normal": c.get("normal", 0), "infra": c.get("infra", 0), "timeout": c.get("timeout", 0)}


def rerun_needed(arm: dict, limit: int = 3) -> list[str]:
    """Rule 9: the tasks to re-run once from the store when an arm has MORE than `limit` infra or timeout ends."""
    bad = sorted((t for t, s in arm["sims"].items() if end_class(s) != "normal"), key=lambda z: int(z) if z.isdigit() else z)
    return bad if len(bad) > limit else []
