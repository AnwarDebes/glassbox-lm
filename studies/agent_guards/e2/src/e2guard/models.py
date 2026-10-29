"""Exact decisions of the learned guards in pure Python (no numpy), from the JSON files that scripts/train_loso.py writes.

Classes are ordered [DENY, ALLOW]; a tie goes to the lowest index (argmax), so a tie DENIES.

Guarded TM (e70's form, DESIGN 1.6):
- learned clauses: the CoTM's non-empty clauses (as dl.cpu_tm_fast.pack keeps them), each a list of literal indices
  over L = [x, 1 - x] (index < F: bit is 1; index >= F: bit F-index is 0), with integer weights [w_deny, w_allow];
- the margin threshold tau as one always-firing clause with weights [0, 1 - tau]: ALLOW wins iff
  s_allow - s_deny >= tau (tau = 1 is the plain argmax);
- one veto clause per written-invariant premise (W1, W2), weights [0, -M] with
  M = (sum of the positive ALLOW weights, bias included) + (sum of |DENY weights|) + 1, so ALLOW cannot win when a veto
  clause fires, whatever the other bits are. The same clause list is what the MILP check encodes.
LR and tree twins: the same bits, the same veto overlay (a premise that holds means DENY), applied in code.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

DENY, ALLOW = 0, 1


def _fires(lits: list[int], x: list[int], F: int) -> bool:
    for l in lits:
        if l < F:
            if not x[l]:
                return False
        elif x[l - F]:
            return False
    return True


class GuardedTM:
    def __init__(self, d: dict) -> None:
        self.d = d
        self.names: list[str] = d["names"]
        self.F = len(self.names)
        self.clauses: list[list[int]] = d["clauses"]
        self.W: list[list[int]] = d["W"]
        self.kind: list[str] = d["clause_kind"]  # learned | bias | veto
        self.veto_id: list[str | None] = d["veto_id"]
        assert len(self.clauses) == len(self.W) == len(self.kind) == len(self.veto_id)

    @classmethod
    def load(cls, path: Path | str) -> GuardedTM:
        return cls(json.loads(Path(path).read_text()))

    def scores(self, x: list[int], with_veto: bool = True, with_bias: bool = True) -> tuple[list[int], list[int]]:
        s = [0, 0]
        fired = []
        for j, (lits, w, k) in enumerate(zip(self.clauses, self.W, self.kind)):
            if (k == "veto" and not with_veto) or (k == "bias" and not with_bias):
                continue
            if _fires(lits, x, self.F):
                s[0] += w[0]
                s[1] += w[1]
                fired.append(j)
        return s, fired

    def decide(self, x: list[int], with_veto: bool = True) -> dict:
        s, fired = self.scores(x, with_veto=with_veto)
        allow = s[ALLOW] > s[DENY]
        vetoes = [self.veto_id[j] for j in fired if self.kind[j] == "veto"]
        learned = [j for j in fired if self.kind[j] == "learned"]
        side = ALLOW if allow else DENY
        top = sorted(learned, key=lambda j: -(self.W[j][side] - self.W[j][1 - side]))[:3]
        return {"allow": allow, "s_deny": s[DENY], "s_allow": s[ALLOW], "margin": s[ALLOW] - s[DENY],
                "vetoes": vetoes, "top_clauses": top}

    def clause_text(self, j: int) -> str:
        parts = []
        for l in self.clauses[j]:
            parts.append(self.names[l] if l < self.F else "NOT " + self.names[l - self.F])
        return " AND ".join(parts) if parts else "(always)"


class LRTwin:
    def __init__(self, d: dict) -> None:
        self.d = d
        self.names = d["names"]
        self.coef = d["coef"]
        self.intercept = d["intercept"]
        self.threshold = d["threshold"]  # P(ALLOW) needed to allow

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def p_allow(self, x: list[int]) -> float:
        z = self.intercept + sum(c for c, v in zip(self.coef, x) if v)
        return 1.0 / (1.0 + math.exp(-z)) if z > -700 else 0.0

    def decide(self, x: list[int]) -> dict:
        p = self.p_allow(x)
        return {"allow": p >= self.threshold, "p_allow": p}


class TreeTwin:
    def __init__(self, d: dict) -> None:
        self.d = d
        self.names = d["names"]
        self.nodes = d["nodes"]  # {"f": bit index, "l": node if bit 0, "r": node if bit 1} or {"p": P(ALLOW)}
        self.threshold = d["threshold"]

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def p_allow(self, x: list[int]) -> float:
        n = self.nodes[0]
        while "p" not in n:
            n = self.nodes[n["r"] if x[n["f"]] else n["l"]]
        return n["p"]

    def decide(self, x: list[int]) -> dict:
        p = self.p_allow(x)
        return {"allow": p >= self.threshold, "p_allow": p}


class RuleSet:
    """D6 (ActGov-style): allow-rules, each a conjunction of literals over the bit names ("name" or "!name").
    ALLOW iff some rule fires; default DENY."""

    def __init__(self, d: dict) -> None:
        self.d = d
        self.names = d["names"]
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.rules = d["rules"]

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def literals(self, rule: list[str]) -> list[int]:
        F = len(self.names)
        out = []
        for lit in rule:
            neg = lit.startswith("!")
            i = self.idx[lit[1:] if neg else lit]
            out.append(i + F if neg else i)
        return out

    def decide(self, x: list[int]) -> dict:
        F = len(self.names)
        for k, rule in enumerate(self.rules):
            if _fires(self.literals(rule), x, F):
                return {"allow": True, "rule": k}
        return {"allow": False, "rule": None}
