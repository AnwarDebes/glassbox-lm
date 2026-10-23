#!/usr/bin/env python
"""e71 v2 report (read only): per rule group, the states certified or refuted at k = 1 and k = 2 (exhaustive search), the
k = 3 and 5 MILP outcomes, and the states whose latest customer utterance is outside the last 3 turns, kept separate.

    report_e71.py --tag e71v2 --seed 1 [--k35-tag e71v2_k35]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "explore" / "verify_e71.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="e71v2")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--k35-tag", default="e71v2_k35")
    args = ap.parse_args()
    rows = [json.loads(l) for l in OUT.read_text().splitlines()]
    en = [d for d in rows if d.get("tag") == args.tag and d.get("kind") == "enum" and d.get("seed") == args.seed]
    k35 = {(d["state"], d["group"], d["budget"]): d for d in rows if d.get("tag") == args.k35_tag and d.get("kind") == "milp_k"
           and d.get("seed") == args.seed}
    out = {"tag": args.tag, "seed": args.seed, "rows": len(en), "states": len({d["state"] for d in en})}
    for window in (True, False):
        sub = [d for d in en if d.get("in_recent") == window]
        per = defaultdict(Counter)
        for d in sub:
            g = d["group"].split("(")[0] if d["group"] == "P_first" else d["group"]
            per[g]["groups"] += 1
            per[g][d["outcome"]] += 1
            if "k1" in d:
                per[g]["k1_breaking_edit_sets"] += d["k1"]["breaking"]
                per[g]["k1_replays_reproduced"] += sum(e.get("replay_breaks_rule", False) for e in d["k1"]["examples"])
                per[g]["k1_replays"] += len(d["k1"]["examples"])
            if "k2" in d:
                per[g]["k2_breaking_edit_sets"] += d["k2"]["breaking"]
                per[g]["k2_replays_reproduced"] += sum(e.get("replay_breaks_rule", False) for e in d["k2"]["examples"])
                per[g]["k2_replays"] += len(d["k2"]["examples"])
            for k in (3, 5):
                r = k35.get((d["state"], d["group"], k))
                if r is not None:
                    per[g][f"k{k}: {r['outcome']}"] += 1
        # state level: a state is certified at k if every one of its groups is (skipped groups count as not certified)
        by_state = defaultdict(list)
        for d in sub:
            by_state[d["state"]].append(d["outcome"])
        st = Counter()
        for s_, outs in by_state.items():
            st["states"] += 1
            st["certified at k = 1 (all groups)"] += all(o in ("certified at k = 2", "certified at k = 1; a 2-word edit breaks the rule") for o in outs)
            st["certified at k = 2 (all groups)"] += all(o == "certified at k = 2" for o in outs)
            st["some group: own choice breaks the rule"] += any(o == "own choice breaks the rule" for o in outs)
        out["in_last_3_turns" if window else "outside_last_3_turns"] = {"per_group": {g: dict(c) for g, c in sorted(per.items())}, "state_level": dict(st)}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
