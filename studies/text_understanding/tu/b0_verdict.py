#!/usr/bin/env python
"""Step 0b, the verdict (PLAN_text_understanding.md section 1b; bars fixed there before any run; VAL only). Reads
tu/b0_picks.json and the seed rows (tag "0b-seeds", seeds 4-6) of tu/results_0b_val.jsonl; writes tu/b0_verdict.json
and .txt (refuses to overwrite). Per set and view: pick minus s 8 (mean of seeds 4-6, in points) and the three paired
differences; "tuning helps" = mean at least +0.5 and all 3 paired differences positive. Per set: the C4-potion gap
(potion minus C4 table, means of seeds 4-6) at s 8 and at the picks; "tuning closes the gap" = gap at the picks at most
half the gap at s 8 (reported only when the gap at s 8 is positive); the share of the gap closed = 1 - pick gap / s-8 gap."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TABLES = {"c4": "ppw_S1000_d500_e0.5", "potion": "potion-base-32M"}
SETS = ["clinc150", "banking77", "hwu64"]
SEEDS = [4, 5, 6]


def main() -> None:
    oj, ot = HERE / "b0_verdict.json", HERE / "b0_verdict.txt"
    assert not oj.exists() and not ot.exists()
    picks = json.loads((HERE / "b0_picks.json").read_text())
    rows = [json.loads(l) for l in open(HERE / "results_0b_val.jsonl")]
    rows = [r for r in rows if r.get("tag") == "0b-seeds" and r["view"] == "existsT"]

    def acc(t, d, s):
        v = []
        for sd in SEEDS:
            m = [r for r in rows if r["table"] == t and r["dataset"] == d and float(r["s"]) == s and r["seed"] == sd]
            assert len(m) == 1, (t, d, s, sd, len(m))
            v.append(m[0]["val_acc"] * 100)
        return np.array(v)

    res, lines = {}, []
    for d in SETS:
        res[d] = {}
        for k, t in TABLES.items():
            p = picks[f"{t}|{d}"]["pick"]
            a8, ap = acc(t, d, 8.0), acc(t, d, p)
            diff = ap - a8
            helps = bool(diff.mean() >= 0.5 - 1e-9 and (diff > 0).all())
            res[d][k] = {"pick": p, "s8_mean": round(float(a8.mean()), 2), "pick_mean": round(float(ap.mean()), 2),
                         "paired": [round(float(x), 2) for x in diff], "gain": round(float(diff.mean()), 2), "helps": helps}
        # gaps from the unrounded means (amended 2026-10-20 before any run: the rounded means could flip an edge case)
        g8 = float(acc(TABLES["potion"], d, 8.0).mean() - acc(TABLES["c4"], d, 8.0).mean())
        gp = float(acc(TABLES["potion"], d, res[d]["potion"]["pick"]).mean() - acc(TABLES["c4"], d, res[d]["c4"]["pick"]).mean())
        res[d]["gap_s8"], res[d]["gap_picks"] = round(g8, 2), round(gp, 2)
        res[d]["closes"] = bool(g8 > 0 and gp <= 0.5 * g8 + 1e-9) if g8 > 0 else None
        res[d]["share_closed"] = round(1 - gp / g8, 3) if g8 > 0 else None
        lines.append(f"{d}: C4 table s{res[d]['c4']['pick']:g} {res[d]['c4']['pick_mean']:.2f} vs s8 {res[d]['c4']['s8_mean']:.2f} "
                     f"(gain {res[d]['c4']['gain']:+.2f}, paired {res[d]['c4']['paired']}, helps {res[d]['c4']['helps']}); "
                     f"potion s{res[d]['potion']['pick']:g} {res[d]['potion']['pick_mean']:.2f} vs s8 {res[d]['potion']['s8_mean']:.2f} "
                     f"(gain {res[d]['potion']['gain']:+.2f}, paired {res[d]['potion']['paired']}, helps {res[d]['potion']['helps']}); "
                     f"gap s8 {g8:.2f} -> picks {gp:.2f}, closes {res[d]['closes']}, share {res[d]['share_closed']}")
    n_help = {k: sum(res[d][k]["helps"] for d in SETS) for k in TABLES}
    n_close = sum(bool(res[d]["closes"]) for d in SETS)
    res["summary"] = {"helps_sets": n_help, "closes_sets": n_close}
    lines.append(f"tuning helps: C4 table on {n_help['c4']} of 3 sets, potion on {n_help['potion']} of 3; "
                 f"tuning closes the gap on {n_close} of 3 sets")
    oj.write_text(json.dumps(res, indent=1))
    ot.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
