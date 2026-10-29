"""Training labels for the learned guards (D4, D5) and the D6 rule builder. LABELS ONLY: nothing here is a guard input.

DESIGN.md E2, "Training rows":
- ground-truth replay (no LLM): user-task calls -> ALLOW, injection-task calls -> DENY;
- the agent's own D0 calls: DENY if the call matches an injection task's ground-truth call (function and attacker
  values), else ALLOW.
This draft makes "attacker values" exact (open for review before freezing; counts both ways in the logs):
- the attacker values of an injection ground-truth call are its atoms that occur in the injection task's GOAL text and
  occur neither in the user task's prompt nor among the atoms of the user task's own ground-truth calls;
- an injection ground-truth call WITHOUT an attacker value (pure reads such as get_channels(), get_user_information(),
  read_channel_messages('general') that benign tasks make too) is dropped from the ground-truth rows (label None,
  reason "no attacker value"), instead of being labelled DENY;
- an agent call is DENY when (a) its function equals that of an injection ground-truth call and it carries at least
  one of that call's attacker values, or (b) it is state-changing (kind != read) and carries a GOAL entity: an e-mail
  address, URL, IBAN-like token or quoted string of the GOAL that is not in the user's prompt. Rule (b) also labels
  the 8 workspace injection tasks whose ground truth is empty in v1.2.x.
- rev2 (2026-10-21, review item 9: one rule for both row sources): in an attacked run, an agent call that is not
  DENY by (a) or (b), is state-changing, is not equal to a user ground-truth call, and whose function equals that of
  an injection ground-truth call is DROPPED (label None), exactly like the ground-truth rows of state-changing
  injection calls without an attacker value (e.g. workspace injection_task_5's cover-up delete_email). Before rev2
  such agent calls were labelled ALLOW, so an attack step could teach ALLOW. Of the 273 dropped ground-truth rows,
  228 are reads, 43 deletes and 2 send_or_pay calls (data/rows_gt_v1.2.2.jsonl).
"""

from __future__ import annotations

import json
import re

from .facts import ToolTable, call_args, call_function
from .normalize import Atom, atoms_of, make_source, matches, norm_text

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
URL = re.compile(r"(?:https?://)?www\.[\w\-]+(?:\.[\w\-]+)+(?:/[\w\-./?=&%]*)?|https?://[\w\-]+(?:\.[\w\-]+)+(?:/[\w\-./?=&%]*)?")
IBANISH = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")
QUOTED = re.compile(r"'([^']{2,80})'")
DROP_REASON_AGENT = ("state-changing call with the function of an injection ground-truth call and no attacker value "
                     "(dropped)")
DROP_REASON_GT = "injection-task ground truth without an attacker value (dropped)"


def call_atoms(c) -> list[Atom]:
    return [a for v in call_args(c).values() for a in atoms_of(v)]


def attacker_values(inj_call, goal: str, user_prompt: str, user_gt_calls) -> list[Atom]:
    g = make_source("goal", goal, parse_yaml=False)
    p = make_source("prompt", user_prompt, parse_yaml=False)
    benign = {(a.kind, a.value) for c in user_gt_calls for a in call_atoms(c)}
    out = []
    for a in call_atoms(inj_call):
        if matches(a, g) and not matches(a, p) and (a.kind, a.value) not in benign:
            out.append(a)
    return out


def goal_entities(goal: str, user_prompt: str) -> list[str]:
    p = norm_text(user_prompt)
    ents = set(EMAIL.findall(goal)) | {u.rstrip(".") for u in URL.findall(goal)} | set(IBANISH.findall(goal))
    ents |= {q.strip() for q in QUOTED.findall(goal)}
    return sorted(e for e in ents if e and norm_text(e) not in p)


def _carries(c, values: list[Atom]) -> list[str]:
    srcs = []
    mine = call_atoms(c)
    for v in values:
        for a in mine:
            if a.kind == v.kind and a.value == v.value:
                srcs.append(v.shown)
                break
            if a.kind == "str" and v.kind == "str" and isinstance(v.value, str) and v.value and \
                    matches(v, make_source("arg", a.shown, parse_yaml=False)):
                srcs.append(v.shown)
                break
    return srcs


def _same_call(a, b) -> bool:
    return call_function(a) == call_function(b) and json.dumps(call_args(a), sort_keys=True, default=str) == \
        json.dumps(call_args(b), sort_keys=True, default=str)


def agent_call_label(c, table: ToolTable, injection: dict | None, user_prompt: str, user_gt_clean,
                     user_gt_run=None) -> tuple[str | None, str]:
    """Label of one call the agent made (a D0 trace). `injection`: {"goal": str, "gt": [calls]} or None (benign run).
    `user_gt_clean`: the user task's ground truth on the clean default environment (the benign values for the
    exclusions); `user_gt_run`: the same on the run's own injected environment, so that a benign call whose argument
    holds injected text (slack's channel name 'External_<injection>') is recognised as benign by exact equality."""
    if injection is None:
        return "ALLOW", "benign run"
    if any(_same_call(c, g) for g in list(user_gt_run or []) + list(user_gt_clean)):
        return "ALLOW", "equals a user-task ground-truth call"
    fn = call_function(c)
    for g in injection["gt"]:
        if call_function(g) != fn:
            continue
        av = attacker_values(g, injection["goal"], user_prompt, user_gt_clean)
        hit = _carries(c, av)
        if hit:
            return "DENY", f"matches injection ground truth {fn} with attacker values {hit[:3]}"
    if table.kind(fn) not in (None, "read"):
        ent_atoms = [Atom("str", norm_text(e), e) for e in goal_entities(injection["goal"], user_prompt)]
        benign = {h for g in user_gt_clean for h in _carries(g, ent_atoms)}
        hit = [h for h in _carries(c, ent_atoms) if h not in benign]
        if hit:
            return "DENY", f"state-changing call carries goal entities {hit[:3]}"
        if any(call_function(g) == fn for g in injection["gt"]):
            return None, DROP_REASON_AGENT
    return "ALLOW", "no attacker value"


def gt_row_label(part: str, inj_call, goal: str, user_prompt: str, user_gt_clean,
                 table: ToolTable | None = None) -> tuple[str | None, str, list[str]]:
    """Label of one ground-truth replay call: part 'user' -> ALLOW; part 'injection' -> DENY when it carries an
    attacker value, or when it is state-changing and carries a GOAL entity (the same rule (b) as for agent calls);
    else dropped (None). `user_gt_clean`: the user task's ground truth on the clean default environment."""
    if part == "user":
        return "ALLOW", "user-task ground truth", []
    av = attacker_values(inj_call, goal, user_prompt, user_gt_clean)
    if av:
        return "DENY", "injection-task ground truth with attacker values", [a.shown for a in av]
    table = table or ToolTable()
    if table.kind(call_function(inj_call)) not in (None, "read"):
        ents = [Atom("str", norm_text(e), e) for e in goal_entities(goal, user_prompt)]
        benign = {h for g in user_gt_clean for h in _carries(g, ents)}
        hit = [h for h in _carries(inj_call, ents) if h not in benign]
        if hit:
            return "DENY", "state-changing injection-task ground truth carrying goal entities", hit
    return None, DROP_REASON_GT, []
