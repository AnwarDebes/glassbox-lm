#!/usr/bin/env python
"""R3 on the 3c stack, the pick (tu/r3c/NOTE_2026-10-22_R3c.md): tu/r3_select.py (e2dff9593ff72773) with these changes and
nothing else: it reads the tag "r3c-screen" rows of tu/r3c/results_r3c_val.jsonl and writes tu/r3c/r3c_picks.json. Docstring of
r3_select.py:

Text understanding, R3 after 3a, the pick: tu/b0_select.py (87a5c2d5e3d36fbf) with these changes and nothing else: it
reads the tag "r3-screen" rows of tu/r3/results_r3_val.jsonl, for the one table a3B_d768, and writes
tu/r3/r3_picks.json; the s-8 reference comparison is dropped (a3B has no earlier seed-1 rows). Docstring of
b0_select.py:

Step 0b, the pick (PLAN_text_understanding.md section 1b, e74's rule; VAL only). Reads the seed-1 screen rows
(tag "0b-screen") of tu/results_0b_val.jsonl and writes tu/b0_picks.json (refuses to overwrite). Per table and set:
the s with the highest val accuracy; if it is less than 0.3 points above s 8, the pick is s 8; an exact tie goes to the
s listed first (2, 4, 8, 12, 16, 24). Accuracies are compared in integer units of 0.01 points (rows keep 4 decimals).
Also reported (descriptive): whether the screen's s-8 seed-1 rows equal the reference seed-1 rows
(results_confirm.jsonl for potion, w1/results_w1_val.jsonl W1-S rows for the C4 table)."""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EMB = HERE.parent
S_LIST = [2.0, 4.0, 8.0, 12.0, 16.0, 24.0]
TABLES = ["a3B_d768"]
SETS = ["clinc150", "banking77", "hwu64"]
REF = {("potion-base-32M", "clinc150"): 0.9443, ("potion-base-32M", "banking77"): 0.9153, ("potion-base-32M", "hwu64"): 0.9099,
       ("ppw_S1000_d500_e0.5", "clinc150"): 0.922, ("ppw_S1000_d500_e0.5", "banking77"): 0.8804, ("ppw_S1000_d500_e0.5", "hwu64"): 0.8903}


def main() -> None:
    out = HERE / "r3c" / "r3c_picks.json"
    assert not out.exists(), f"{out} exists"
    rows = [json.loads(l) for l in open(HERE / "r3c" / "results_r3c_val.jsonl")]
    rows = [r for r in rows if r.get("tag") == "r3c-screen" and r["seed"] == 1 and r["view"] == "existsT"]
    res = {}
    for t in TABLES:
        for d in SETS:
            acc = {}
            for s in S_LIST:
                m = [r for r in rows if r["table"] == t and r["dataset"] == d and float(r["s"]) == s]
                assert len(m) == 1, (t, d, s, len(m))
                acc[s] = round(m[0]["val_acc"] * 10000)
            best = S_LIST[0]
            for s in S_LIST[1:]:
                if acc[s] > acc[best]:
                    best = s
            pick = best if acc[best] - acc[8.0] >= 30 else 8.0
            res[f"{t}|{d}"] = {"table": t, "dataset": d, "pick": pick, "best": best,
                               "screen_val_pct": {str(s): acc[s] / 100 for s in S_LIST}}
    out.write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(k, "pick", v["pick"], "best", v["best"], v["screen_val_pct"])


if __name__ == "__main__":
    main()
