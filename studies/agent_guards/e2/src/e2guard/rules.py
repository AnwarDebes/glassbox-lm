"""Deterministic arms of E2 over the trusted facts: the written invariants W1-W2, D2 (written deny rules), D3 (the
ROPE-style origin gate as DESIGN.md states it) and D3r (the same gate on ROPE's own sensitive-parameter table).

Every function takes the Facts of one call (facts.py), the fact bits that are on (bits.py) and the trusted query, and
returns (allow, reason, rule ids). Unknown tools are allowed in every arm: the runtime rejects them itself, so every arm
sees the same error message as the no-defence arm.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .bits import vetoes_fired
from .facts import Facts, ToolTable, sha256_file

HERE = Path(__file__).resolve().parent
D2_PATH = HERE / "d2_rules.json"
TRUSTED = {"user_query", "named_source"}
WORD = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def invariants(on: set[str]) -> list[str]:
    """Ids of the W1/W2 premises that hold (each one means: not ALLOW)."""
    return vetoes_fired(on)


def explain_atoms(f: Facts, roles=None, allowed=TRUSTED, params=None) -> list[str]:
    out = []
    for a in f.args:
        if roles is not None and a.role not in roles:
            continue
        if params is not None and a.name not in params:
            continue
        for t in a.atoms:
            if t.prov not in allowed:
                out.append(f"{a.name}={t.shown[:60]!r} ({t.prov})")
    return out


def decide_w(f: Facts, on: set[str]) -> tuple[bool, str, list[str]]:
    fired = invariants(on)
    if fired:
        return False, "written invariant " + ", ".join(fired), fired
    return True, "no invariant fires", []


class D2Rules:
    def __init__(self, path: Path | str = D2_PATH) -> None:
        self.path = Path(path)
        d = json.loads(self.path.read_text())
        self.words = {k: set(v) for k, v in d["action_words"].items()}
        self.sha256 = sha256_file(self.path)

    def decide(self, f: Facts, on: set[str], query: str) -> tuple[bool, str, list[str]]:
        if f.kind is None:
            return True, "unknown tool (the runtime rejects it)", []
        fired = invariants(on)
        if fired:
            return False, "written invariant " + ", ".join(fired), fired
        if f.kind == "read":
            return True, "read call", []
        words = set(WORD.findall(query.casefold()))
        need = self.words.get(f.function)
        if need is None:
            return False, f"D2: no rule allows the state-changing tool {f.function}", ["D2:no_rule"]
        if words & need:
            return True, f"D2: the query asks for this action ({sorted(words & need)[:3]})", []
        return False, f"D2: the user's query does not ask for {f.function}", ["D2:not_asked"]


def decide_d3(f: Facts, on: set[str]) -> tuple[bool, str, list[str]]:
    """DESIGN's D3: W1-W2, and for every state-changing tool every data-carrying argument must trace to the user's
    query or to a named source (so D3 is at least as strict as W1-W2)."""
    if f.kind is None:
        return True, "unknown tool (the runtime rejects it)", []
    fired = invariants(on)
    if fired:
        return False, "written invariant " + ", ".join(fired), fired
    if f.kind == "read":
        return True, "read call", []
    bad = explain_atoms(f)
    if bad:
        return False, "origin gate: values that do not trace to your query or a source it named: " + "; ".join(bad[:4]), ["D3"]
    return True, "origin gate: every value traces to the query or a named source", []


class D3Rope:
    """D3r: W1-W2 plus ROPE's sensitive parameters (tool_table.json, rope_scope): 'sourced' values must trace to the
    query or a named source, 'prompt' and 'explicit' values must be literals of the query. Other tools are allowed."""

    def __init__(self, table: ToolTable | None = None) -> None:
        self.table = table or ToolTable()
        scope: dict[str, dict] = {}
        for tools in self.table.rope.values():
            for tool, params in tools.items():
                if tool in scope and scope[tool] != params:
                    # the same tool name in two suites (send_email, create_calendar_event, cancel_calendar_event):
                    # take the union of the guarded parameters, the stricter marker winning
                    merged = dict(scope[tool])
                    for p, mk in params.items():
                        merged[p] = mk if (p not in merged or mk != "sourced") else merged[p]
                    scope[tool] = merged
                else:
                    scope[tool] = dict(params)
        self.scope = scope

    def decide(self, f: Facts, on: set[str]) -> tuple[bool, str, list[str]]:
        if f.kind is None:
            return True, "unknown tool (the runtime rejects it)", []
        fired = invariants(on)
        if fired:
            return False, "written invariant " + ", ".join(fired), fired
        params = self.scope.get(f.function, {})
        bad = []
        for name, marker in params.items():
            allowed = TRUSTED if marker == "sourced" else {"user_query"}
            bad += explain_atoms(f, allowed=allowed, params={name})
        if bad:
            return False, "origin gate (ROPE scope): " + "; ".join(bad[:4]), ["D3r"]
        return True, "origin gate (ROPE scope): passes", []
