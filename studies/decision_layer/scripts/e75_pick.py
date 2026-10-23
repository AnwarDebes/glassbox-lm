#!/usr/bin/env python
"""e75 picks (plan logs/plan_e75_abcd_s.md): the TM's s on ABCD dev for the C4 agent and the bge-large agent.

    e75_pick.py      # prints both screens and both picks

Rule (fixed in the plan), for each agent: the highest 4k dev score among s 8, 12, 16 and 24; it must be at least 0.3
points above that agent's own s 8 row, else s 8; an exact tie goes to the first of [8, 12, 16, 24]. Read only.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S = [8, 12, 16, 24]
TAGS = {"C4": {8: "e72_abcd_Bc4_c4000", 12: "e74_abcd_Bc4_c4000_s12", 16: "e74c_abcd_Bc4_c4000_s16", 24: "e75_abcd_Bc4_c4000_s24"},
        "bge-large": {s: f"e75_abcd_bge_c4000_s{s}" for s in S}}
NEW = {"e75_abcd_Bc4_c4000_s24"} | {f"e75_abcd_bge_c4000_s{s}" for s in S}


def main() -> None:
    rows = {}
    for line in (ROOT / "results" / "explore" / "abcd.jsonl").read_text().splitlines():
        d = json.loads(line)
        if d.get("arm") == "tm" and d.get("tag") in {t for m in TAGS.values() for t in m.values()}:
            rows.setdefault(d["tag"], []).append(d)
    frozen = json.loads((ROOT / "logs" / "plan_e75_abcd_s.frozen.json").read_text())["frozen_utc"]
    for agent, tags in TAGS.items():
        acc = {}
        for s in S:
            got = rows.get(tags[s], [])
            if len(got) != 1:
                raise SystemExit(f"{tags[s]}: {len(got)} rows (need exactly 1)")
            d = got[0]
            assert d["dataset"] == "abcd" and d["cfg"]["s"] == float(s) and d["cfg"]["n_clauses"] == 4000 and d["cfg"]["T"] == 4000 and d["seed"] == 1, d["tag"]
            if d["tag"] in NEW:
                assert d["time"] > frozen, f"{d['tag']} before the freeze"
            acc[s] = d["acc"]
            print(f"{agent} s {s:>2}: dev {d['acc']:.4f} (best epoch {d['best_epoch']}, {d['train_s']} s; {d['tag']})")
        best = max(S, key=lambda s: (acc[s], -S.index(s)))
        p = best if acc[best] - acc[8] >= 0.003 - 1e-12 else 8
        print(f"{agent}: best s {best} ({acc[best]:.4f}); s 8 {acc[8]:.4f}; gap {100 * (acc[best] - acc[8]):+.2f}; pick s {p}")


if __name__ == "__main__":
    main()
