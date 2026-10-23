#!/usr/bin/env python
"""Reports of the registered agent-loop tests R5 (bandit) and R6 (clarify-or-route, TD).

    report_agent.py R5     reads results/confirm/R5/bandit.jsonl
    report_agent.py R6     reads results/confirm/R6/r6.jsonl

Differences between two learners are paired by seed; 95% percentile bootstrap over seeds
(10,000 resamples, seed 0).
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def boot_seeds(a: dict, b: dict, n: int = 10000, seed: int = 0) -> dict:
    ks = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in ks])
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(n, len(d)))].mean(1)
    return {"diff": float(d.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)), "n": len(ks)}


def r5() -> dict:
    rows = [json.loads(l) for l in open(ROOT / "results" / "confirm" / "R5" / "bandit.jsonl")]
    by = collections.defaultdict(dict)  # (task, learner) -> seed -> acc
    cum = collections.defaultdict(dict)
    for r in rows:
        task = f'{r["dataset"]}:{r["target"]}'
        by[(task, r["learner"])][r["seed"]] = r["final_eval_acc"]
        if "cum_online_acc" in r:
            cum[(task, r["learner"])][r["seed"]] = r["cum_online_acc"]
    tasks = sorted({t for t, _ in by})
    out = {"tasks": {}}
    n511 = n52 = 0
    for t in tasks:
        res = {ln: {"mean": float(np.mean(list(v.values()))), "n": len(v)} for (tt, ln), v in by.items() if tt == t}
        for (tt, ln), v in cum.items():
            if tt == t:
                res[ln]["cum_online_mean"] = float(np.mean(list(v.values())))
        base = max(("linucb", "eps_logistic"), key=lambda ln: res.get(ln, {"mean": -1})["mean"])
        cmp = boot_seeds(by[(t, "tm")], by[(t, base)])
        ratio = res["tm"]["mean"] / res["sup_tm"]["mean"]
        ok51 = cmp["diff"] >= -0.010
        ok52 = ratio >= 0.97
        n511 += ok51
        n52 += ok52
        out["tasks"][t] = {"learners": res, "best_baseline": base, "tm_minus_best_baseline": cmp,
                           "tm_over_sup_tm": ratio, "R5.1_task": ok51, "R5.2_task": ok52}
    out["R5.1"] = n511 >= 2
    out["R5.2"] = n52 >= 2
    out["R5_met"] = out["R5.1"] and out["R5.2"]
    return out


def r5b() -> dict:
    rows = [json.loads(l) for l in open(ROOT / "results" / "confirm" / "R5b" / "bandit.jsonl")]
    by = collections.defaultdict(dict)
    cum = collections.defaultdict(dict)
    for r in rows:
        task = f'{r["dataset"]}:{r["target"]}'
        by[(task, r["learner"])][r["seed"]] = r["final_eval_acc"]
        if "cum_online_acc" in r:
            cum[(task, r["learner"])][r["seed"]] = r["cum_online_acc"]
    out = {"tasks": {}}
    ok1 = ok2 = True
    for t in sorted({t for t, _ in by}):
        res = {ln: {"mean": float(np.mean(list(v.values()))), "n": len(v)} for (tt, ln), v in by.items() if tt == t}
        for (tt, ln), v in cum.items():
            if tt == t:
                res[ln]["cum_online_mean"] = float(np.mean(list(v.values())))
        cmp = boot_seeds(by[(t, "tm")], by[(t, "linucb")])
        ratio = res["tm"]["mean"] / res["sup_tm"]["mean"]
        r1 = res["tm"]["mean"] - res["linucb"]["mean"] >= -0.010
        r2 = ratio >= 0.97
        ok1 &= r1
        ok2 &= r2
        out["tasks"][t] = {"learners": res, "tm_minus_linucb": cmp, "tm_over_sup_tm": ratio, "R5b.1_task": r1, "R5b.2_task": r2}
    out["R5b.1"] = bool(ok1)
    out["R5b.2"] = bool(ok2)
    out["R5b_met"] = bool(ok1 and ok2)
    return out


def r5c() -> dict:
    """R5c (CLINC150 intents, 150 actions): R5c.1 TM bandit >= 0.97 sup-TM (= met); R5c.2 TM bandit
    >= LinUCB - 1.0 point, a separate verdict."""
    rows = [json.loads(l) for l in open(ROOT / "results" / "confirm" / "R5c" / "bandit.jsonl")]
    rows = [r for r in rows if r.get("tag") == "R5c"]
    by = collections.defaultdict(dict)
    cum = collections.defaultdict(dict)
    for r in rows:
        by[r["learner"]][r["seed"]] = r["final_eval_acc"]
        if "cum_online_acc" in r:
            cum[r["learner"]][r["seed"]] = r["cum_online_acc"]
    res = {ln: {"mean": float(np.mean(list(v.values()))), "n": len(v), "per_seed": v} for ln, v in by.items()}
    for ln, v in cum.items():
        res[ln]["cum_online_mean"] = float(np.mean(list(v.values())))
    out = {"task": "clinc150:intent", "learners": res}
    if all(k in res for k in ("tm", "linucb", "sup_tm")):
        out["tm_minus_linucb"] = boot_seeds(by["tm"], by["linucb"])
        out["tm_over_sup_tm"] = res["tm"]["mean"] / res["sup_tm"]["mean"]
        out["R5c.1"] = bool(out["tm_over_sup_tm"] >= 0.97)
        out["R5c.2"] = bool(res["tm"]["mean"] - res["linucb"]["mean"] >= -0.010)
        out["R5c_met"] = out["R5c.1"]
    return out


def r5d() -> dict:
    """R5d (Banking77 intents, 77 actions, 30 passes): R5d.1 TM bandit >= LinUCB - 1.0 point and
    R5d.2 TM bandit >= 0.97 sup-TM; met if both (the rules of R5b)."""
    rows = [json.loads(l) for l in open(ROOT / "results" / "confirm" / "R5d" / "bandit.jsonl")]
    rows = [r for r in rows if r.get("tag") == "R5d"]
    by = collections.defaultdict(dict)
    cum = collections.defaultdict(dict)
    for r in rows:
        by[r["learner"]][r["seed"]] = r["final_eval_acc"]
        if "cum_online_acc" in r:
            cum[r["learner"]][r["seed"]] = r["cum_online_acc"]
    res = {ln: {"mean": float(np.mean(list(v.values()))), "n": len(v), "per_seed": v} for ln, v in by.items()}
    for ln, v in cum.items():
        res[ln]["cum_online_mean"] = float(np.mean(list(v.values())))
    out = {"task": "banking77:intent", "learners": res}
    if all(k in res for k in ("tm", "linucb", "sup_tm")):
        out["tm_minus_linucb"] = boot_seeds(by["tm"], by["linucb"])
        out["tm_over_sup_tm"] = res["tm"]["mean"] / res["sup_tm"]["mean"]
        out["R5d.1"] = bool(res["tm"]["mean"] - res["linucb"]["mean"] >= -0.010)
        out["R5d.2"] = bool(out["tm_over_sup_tm"] >= 0.97)
        out["R5d_met"] = out["R5d.1"] and out["R5d.2"]
    return out


def r6(tag: str = "R6") -> dict:
    rows = [json.loads(l) for l in open(ROOT / "results" / "confirm" / "R6" / "r6.jsonl")]
    rows = [r for r in rows if r.get("tag") == tag]  # "R6": registered runs; "R6sens": post-hoc sensitivity
    by = collections.defaultdict(dict)
    extra = collections.defaultdict(list)
    for r in rows:
        name = r["learner"] + ("" if r.get("memory", True) else "_nomem")
        by[name][r["seed"]] = r["eval_result"]["mean_return"]
        extra[name].append(r["eval_result"])
    out = {"learners": {k: {"mean_return": float(np.mean(list(v.values()))), "n": len(v),
                            "share_ask": float(np.mean([e["share_ask"] for e in extra[k]])),
                            "share_route0": float(np.mean([e["share_route0"] for e in extra[k]])),
                            "share_escalate0": float(np.mean([e["share_escalate0"] for e in extra[k]])),
                            "return_inscope": float(np.mean([e["mean_return_inscope"] for e in extra[k]])),
                            "return_oos": float(np.mean([e.get("mean_return_oos", np.nan) for e in extra[k]]))}
                        for k, v in by.items()}}
    if "tm_q" in by and "tm_myopic" in by:
        out["R6.1_tmq_minus_myopic"] = boot_seeds(by["tm_q"], by["tm_myopic"])
        out["R6.1"] = out["R6.1_tmq_minus_myopic"]["diff"] >= 0.02 and out["R6.1_tmq_minus_myopic"]["lo"] > 0
    if "tm_q" in by and "rule" in by:
        out["R6.2_tmq_minus_rule"] = boot_seeds(by["tm_q"], by["rule"])
        out["R6.2"] = out["R6.2_tmq_minus_rule"]["diff"] >= -0.01
    if "tm_q_nomem" in by:
        out["R6.3_tmq_minus_nomem"] = boot_seeds(by["tm_q"], by["tm_q_nomem"])
    if "lin_q" in by:
        out["R6.4_tmq_minus_linq"] = boot_seeds(by["tm_q"], by["lin_q"])
    out["R6_met"] = bool(out.get("R6.1")) and bool(out.get("R6.2"))
    return out


if __name__ == "__main__":
    which = sys.argv[1]
    tag = sys.argv[2] if len(sys.argv) > 2 else which
    res = {"R5": r5, "R5b": r5b, "R5c": r5c, "R5d": r5d}[which]() if which in ("R5", "R5b", "R5c", "R5d") else r6(tag)
    path = ROOT / "results" / "confirm" / which / f"report_{tag}.json"  # report_R6.json or report_R6sens.json
    path.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
