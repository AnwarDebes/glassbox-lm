#!/usr/bin/env python
"""Apply the frozen selection rules of R5 (bandit) and R6 (clarify-or-route) to the exploratory
results, and print the chosen settings with every candidate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def r5() -> dict:
    rows = [json.loads(l) for l in open(ROOT / "results" / "explore" / "bandit.jsonl")]
    rows = [r for r in rows if r["tag"] in ("e3", "e6") and r["dataset"] == "clinc150" and r["target"] == "domain"]
    out = {}
    for fam in ("tm", "linucb", "eps_logistic"):
        cand = [r for r in rows if r["learner"] == fam]
        keys = {"tm": ("eps", "eps_decay", "topk", "replay"), "linucb": ("alpha",), "eps_logistic": ("eps", "topk", "lr", "replay")}[fam]
        listing = [({k: r.get(k) for k in keys}, round(r["final_eval_acc"], 4)) for r in cand]
        best = max(cand, key=lambda r: r["final_eval_acc"]) if cand else None
        out[fam] = {"candidates": listing, "chosen": ({k: best.get(k) for k in keys}, round(best["final_eval_acc"], 4)) if best else None}
    return out


def r6() -> dict:
    rows = [json.loads(l) for l in open(ROOT / "results" / "explore" / "r6.jsonl")]
    rows = [r for r in rows if str(r.get("tag", "")).startswith("e12_")]
    out = {}
    for fam in ("tm_q", "lin_q"):
        cand = [r for r in rows if r["learner"] == fam and r.get("memory", True)]
        keys = ("fqi", "kappa", "updates", "lr", "fqi_iters", "fqi_epochs", "clauses", "T", "s")
        listing = [(r["tag"], {k: r.get(k) for k in keys if k in r}, round(r["val"]["mean_return"], 4)) for r in cand]
        best = max(cand, key=lambda r: r["val"]["mean_return"]) if cand else None
        out[fam] = {"candidates": listing, "chosen": (best["tag"], {k: best.get(k) for k in keys if k in best},
                                                     round(best["val"]["mean_return"], 4)) if best else None}
    for fam in ("rule", "route", "ask_route"):
        c = [r for r in rows if r["learner"] == fam]
        if c:
            out[fam] = round(c[-1]["val"]["mean_return"], 4)
    return out


if __name__ == "__main__":
    which = sys.argv[1]
    print(json.dumps(r5() if which == "R5" else r6(), indent=1))
