"""E2 arms as tool-call policies for harness_build/agentdojo/harness/guard.py (GuardedToolsExecutor, veto mode).

Each policy gets only what a deployed guard sees (query, the proposed call, the messages so far, the tool schemas) and
computes the trusted facts itself (facts.py). Every decision is appended to `self.log` (one dict per call, the
DESIGN 1.11 step contract adapted to tool calls); the driver writes it out.

Arms: D0 AllowAllRecorder (records facts, never blocks), W (W1-W2 only, a diagnostic), D2, D3, D3r, D4 (guarded TM),
D5 (LR twin, tree twin; with the W1-W2 overlay), D6 (ActGov-style rule set; deployed only after the solver check).
D1 (tool_filter) and D7 (firewall) need the LLM and live elsewhere (pipelines_e2.py).
A learned arm holds one model per held-out suite (leave one suite out); the driver calls set_suite() before each run.
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../../.."))

import sys
import time
from pathlib import Path

AGENTDOJO_HARNESS = Path(f"{_REPO_ROOT}/studies/agent_guards/agentdojo/harness")
if str(AGENTDOJO_HARNESS) not in sys.path:
    sys.path.insert(0, str(AGENTDOJO_HARNESS))

from guard import (
    Verdict,
)

from .bits import FACT_BITS, QueryBits, fact_bits_on, vector
from .facts import FactExtractor, ToolTable, call_args, call_function
from .models import GuardedTM, LRTwin, RuleSet, TreeTwin
from .rules import D2Rules, D3Rope, decide_d3, decide_w, invariants


class BasePolicy:
    name = "base"

    def __init__(self, extractor: FactExtractor | None = None) -> None:
        self.x = extractor or FactExtractor(ToolTable())
        self.log: list[dict] = []
        self.suite: str | None = None

    def set_suite(self, suite: str) -> None:
        self.suite = suite

    def judge(self, query, facts, on) -> tuple[bool, str, list[str], dict]:
        raise NotImplementedError

    def decide(self, query, call, messages, runtime) -> Verdict:
        t0 = time.perf_counter()
        f = self.x.extract(query, call, messages)
        if f.kind is None:
            allow, reason, rules, extra = True, "unknown tool (the runtime rejects it)", [], {}
            on: set[str] = set()
        else:
            on = fact_bits_on(f)
            allow, reason, rules, extra = self.judge(query, f, on)
        ms = round((time.perf_counter() - t0) * 1000, 3)
        self.log.append({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "policy": self.name,
                         "suite": self.suite, "function": call_function(call), "args": call_args(call),
                         "facts": f.to_json(), "facts_on": sorted(on),
                         "verdict": {"v": "ALLOW" if allow else "DENY", "rules": rules, **extra}, "ms": ms})
        score = extra.get("margin", extra.get("p_allow"))
        return Verdict(allow, reason, None if score is None else float(score))


class AllowAllRecorder(BasePolicy):
    """D0 in the guarded pipeline: never blocks; records the facts (training rows, B5-style identity checks)."""

    name = "d0_record"

    def judge(self, query, f, on):
        return True, "allow (recorder)", [], {"would_veto": invariants(on)}


class WOnly(BasePolicy):
    name = "w_only"

    def judge(self, query, f, on):
        allow, reason, rules = decide_w(f, on)
        return allow, reason, rules, {}


class D2Policy(BasePolicy):
    name = "d2_rules"

    def __init__(self, extractor=None, rules: D2Rules | None = None) -> None:
        super().__init__(extractor)
        self.rules = rules or D2Rules()

    def judge(self, query, f, on):
        allow, reason, rules = self.rules.decide(f, on, query)
        return allow, reason, rules, {}


class D3Policy(BasePolicy):
    name = "d3_origin"

    def judge(self, query, f, on):
        allow, reason, rules = decide_d3(f, on)
        return allow, reason, rules, {}


class D3RopePolicy(BasePolicy):
    name = "d3r_rope_scope"

    def __init__(self, extractor=None) -> None:
        super().__init__(extractor)
        self.gate = D3Rope(self.x.table)

    def judge(self, query, f, on):
        allow, reason, rules = self.gate.decide(f, on)
        return allow, reason, rules, {}


class FoldModelPolicy(BasePolicy):
    """A learned guard with one model and one query-bit table per held-out suite: {suite: (model_json, qbits_json)}."""

    model_cls = None

    def __init__(self, paths: dict[str, tuple[str, str]], extractor=None) -> None:
        super().__init__(extractor)
        self.models = {s: self.model_cls.load(m) for s, (m, _) in paths.items()}
        self.qbits = {s: QueryBits(q) for s, (_, q) in paths.items()}
        for s, m in self.models.items():
            assert m.names == FACT_BITS + self.qbits[s].names, f"bit names of the {s} model differ from the extractor's"

    def x_of(self, f, on) -> list[int]:
        assert self.suite is not None, "set_suite() first"
        q = self.qbits[self.suite].on(f.query_sha256)
        return vector(self.models[self.suite].names, on | q)


class D4Policy(FoldModelPolicy):
    """The TM guard: the CoTM's clauses + tau bias + W1-W2 veto clauses, one exact integer decision."""

    name = "d4_tm"
    model_cls = GuardedTM

    def judge(self, query, f, on):
        r = self.models[self.suite].decide(self.x_of(f, on))
        if r["vetoes"]:
            reason = "written invariant " + ", ".join(r["vetoes"])
        elif r["allow"]:
            reason = f"TM guard allows (margin {r['margin']})"
        else:
            reason = f"TM guard denies (margin {r['margin']}; clauses {r['top_clauses']})"
        return r["allow"], reason, list(r["vetoes"]), {"margin": r["margin"], "top_clauses": r["top_clauses"]}


class _OverlayTwin(FoldModelPolicy):
    def judge(self, query, f, on):
        fired = invariants(on)
        r = self.models[self.suite].decide(self.x_of(f, on))
        if fired:
            return False, "written invariant " + ", ".join(fired), fired, {"p_allow": r["p_allow"]}
        reason = f"{self.name} {'allows' if r['allow'] else 'denies'} (P(allow) {r['p_allow']:.3f})"
        return r["allow"], reason, [], {"p_allow": r["p_allow"]}


class D5LRPolicy(_OverlayTwin):
    name = "d5_lr"
    model_cls = LRTwin


class D5TreePolicy(_OverlayTwin):
    name = "d5_tree"
    model_cls = TreeTwin


class D6Policy(FoldModelPolicy):
    """ActGov-style allow-rules (default deny), used only if the solver check of W1-W2 passed for that fold."""

    name = "d6_rules"
    model_cls = RuleSet

    def __init__(self, paths, extractor=None) -> None:
        super().__init__(paths, extractor)
        for s, m in self.models.items():
            assert m.d.get("solver_check", {}).get("passed") is True, f"D6 rule set for {s} has no passed solver check"

    def judge(self, query, f, on):
        r = self.models[self.suite].decide(self.x_of(f, on))
        if r["allow"]:
            return True, f"allow-rule {r['rule']} matches", [], {"rule": r["rule"]}
        return False, "no allow-rule matches this call", ["D6:no_rule"], {"rule": None}
