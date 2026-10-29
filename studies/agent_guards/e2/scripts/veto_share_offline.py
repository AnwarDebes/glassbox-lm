"""The budget rule's fallback veto shares (registration rev3, section 11; the decision 1): per replay arm, the
share of Appendix B's offline ground-truth replay episodes (97 benign + 949 attacked, scripted always-fooled agent) in
which the arm vetoes at least one call. Used by report_e2.py budget only for an arm that E2-0's open-loop decide()
check did not cover. Pure Python (either venv), CPU only, no LLM:

    python -I -B scripts/veto_share_offline.py --out data/pins/veto_share_offline.json

Exact without a pipeline: the rows (data/rows_gt_v1.2.2.jsonl) hold every call of every replayed episode with the
facts the recorder saw; an arm sees the same history until its first veto, so "at least one veto in the episode" is
"some row of the episode is vetoed". Unknown tools are allowed by every arm. The learned arms (D4, D5, D6) use the
stand-in models (models/standin): the final models do not exist before the build stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

E2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(E2 / "src"))

from e2guard.bits import QueryBits, vector
from e2guard.facts import ToolTable, facts_from_json
from e2guard.models import GuardedTM, LRTwin, RuleSet, TreeTwin
from e2guard.rules import D2Rules, D3Rope, decide_d3, decide_w, invariants

SUITES = ["banking", "slack", "travel", "workspace"]
ARMS = ["w_only", "d2", "d3", "d3r", "d4_tm", "d5_lr", "d5_tree", "d6_rules"]
COUNTS = {"banking": (16, 144), "slack": (21, 105), "travel": (20, 140), "workspace": (40, 560)}  # benign, pairs


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class Deciders:
    def __init__(self, models: Path, qbits: Path) -> None:
        self.t = ToolTable()
        self.d2, self.d3r = D2Rules(), D3Rope(self.t)
        self.m = {(a, s): cls.load(models / f"{a}_{s}.json") for s in SUITES
                  for a, cls in (("d4_tm", GuardedTM), ("d5_lr", LRTwin), ("d5_tree", TreeTwin), ("d6_rules", RuleSet))}
        for s in SUITES:
            assert self.m[("d6_rules", s)].d.get("solver_check", {}).get("passed") is True
        self.qb = {s: QueryBits(qbits / f"fold_{s}.json") for s in SUITES}

    def allow(self, arm: str, row: dict, query: str) -> bool:
        if row["facts"]["kind"] is None:
            return True  # unknown tool: every arm allows it (the runtime rejects it)
        f, on = facts_from_json(row["facts"]), set(row["facts_on"])
        if arm == "w_only":
            return decide_w(f, on)[0]
        if arm == "d2":
            return self.d2.decide(f, on, query)[0]
        if arm == "d3":
            return decide_d3(f, on)[0]
        if arm == "d3r":
            return self.d3r.decide(f, on)[0]
        s = row["suite"]  # the model whose held-out suite this is
        m = self.m[(arm, s)]
        x = vector(m.names, on | self.qb[s].on(row["query_sha256"]))
        if arm == "d4_tm" or arm == "d6_rules":
            return bool(m.decide(x)["allow"])
        return (not invariants(on)) and bool(m.decide(x)["allow"])  # D5: the W1-W2 overlay, then the twin


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=str(E2 / "data" / "rows_gt_v1.2.2.jsonl"))
    ap.add_argument("--models", default=str(E2 / "models" / "standin"))
    ap.add_argument("--qbits", default=str(E2 / "data" / "query_bits"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    assert not out.exists(), f"{out} exists"
    tasks = json.loads((E2 / "data" / "tasks_v1.2.2.json").read_text())
    dec = Deciders(Path(a.models), Path(a.qbits))
    vetoed: dict = defaultdict(set)
    episodes: set = set()
    for line in Path(a.rows).read_text().splitlines():
        r = json.loads(line)
        ep = (r["suite"], r["user_task"], r["injection_task"])
        episodes.add(ep)
        q = tasks[r["suite"]]["user_tasks"][r["user_task"]]["prompt"]
        for arm in ARMS:
            if not dec.allow(arm, r, q):
                vetoed[arm].add(ep)
    want = sum(b + p for b, p in COUNTS.values())
    assert len(episodes) == want, f"{len(episodes)} episodes with rows, {want} expected"
    res = {"note": ("Appendix B's offline ground-truth replay (scripted always-fooled agent; stand-in models for D4, D5, "
                    "D6): per arm, the episodes with at least one vetoed call; the budget rule's fallback only"),
           "rows_sha256": sha256(Path(a.rows)), "models": str(Path(a.models).resolve().relative_to(E2)),
           "episodes": want, "arms": {}}
    for arm in ARMS:
        v = vetoed[arm]
        per = {s: {"vetoed": sum(1 for e in v if e[0] == s), "episodes": sum(COUNTS[s])} for s in SUITES}
        ben = sum(1 for e in v if e[2] is None)
        res["arms"][arm] = {"vetoed": len(v), "share": len(v) / want, "per_suite": per,
                            "benign": {"vetoed": ben, "episodes": 97}, "attacked": {"vetoed": len(v) - ben, "episodes": 949}}
    out.write_text(json.dumps(res, indent=1) + "\n")
    for arm, d in res["arms"].items():
        print(f"{arm:9} {d['vetoed']:5}/{want} = {d['share']:.4f}  benign {d['benign']['vetoed']}/97  "
              f"attacked {d['attacked']['vetoed']}/949")


if __name__ == "__main__":
    main()
