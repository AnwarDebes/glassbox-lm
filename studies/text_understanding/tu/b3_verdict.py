#!/usr/bin/env python
"""Step 3b, the verdict (tu/PLAN_3b.md, "Bar"), from the rows of b3_exists_tm.py and b3_lr.py.

    python -I -B b3_verdict.py [--b3 tu/b3]   # reads tu/b3/base.json, results_b3_val.jsonl, results_b3_lr.jsonl

- Reference rows: base.json "ref_tm" (its tag, table = base.json "base", s = the base's s, seeds 4-6) and "ref_lr"
  (table = the base); arms: TM rows of tag "3b-tm" with arm 3b-T, 3b-L, 3b-I, 3b-C or 3b-ceil at the base's s, seeds 4-6;
  LR rows named by the arm.
- Gains in points, unrounded; TM = the mean over seeds 4-6 of the seed-paired differences; LR = val.
- PASS when, on at least 2 of the 3 sets, all of these hold with both probes: 3b-T - base >= 0.5 - 1e-9;
  3b-T - 3b-L >= 0.3 - 1e-9; 3b-T - 3b-I >= 0.3 - 1e-9. Otherwise FAIL, with the reason (part 1 missing, or the gain
  is a linear re-encoding, or it comes from the second set of concept bits).
- Descriptive: every arm's gains over base; the seed SD of each arm's TM rows; the SE of each paired TM difference
  (SD of the 3 differences / sqrt 3); the share of potion's TM gap closed (base.json "potion_tm"); the on-shares.
Writes b3/verdict_b3.json and b3/verdict_b3.txt; refuses to overwrite either.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SETS = ("clinc150", "banking77", "hwu64")
SEEDS = (4, 5, 6)
ARMS = ("3b-T", "3b-L", "3b-I", "3b-C", "3b-ceil")
BAR_BASE, BAR_TWIN = 0.5 - 1e-9, 0.3 - 1e-9


def jsonl(p: Path) -> list[dict]:
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


def one(rows: list[dict], what: str, **cond) -> dict:
    m = [r for r in rows if all(r.get(k) == v for k, v in cond.items())]
    if len(m) != 1:
        raise SystemExit(f"{what}: {len(m)} rows match {cond}")
    return m[0]


def per_seed(rows, **cond) -> list[float]:
    return [100.0 * float(one(rows, f"seed {sd}", seed=sd, **cond)["val_acc"]) for sd in SEEDS]


def paired(a: list[float], b: list[float]) -> dict:
    d = [x - y for x, y in zip(a, b)]
    return {"mean": st.mean(d), "per_seed": d, "se": st.stdev(d) / math.sqrt(len(d))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--b3", type=Path, default=HERE / "b3")
    a = ap.parse_args()
    oj, ot = a.b3 / "verdict_b3.json", a.b3 / "verdict_b3.txt"
    for p in (oj, ot):
        if p.exists():
            print(f"refusing to overwrite {p}", file=sys.stderr)
            return 2
    base = json.loads((a.b3 / "base.json").read_text())
    val, lr = jsonl(a.b3 / "results_b3_val.jsonl"), jsonl(a.b3 / "results_b3_lr.jsonl")
    ref = [r for r in jsonl(Path(base["ref_tm"]["file"])) if r.get("tag") == base["ref_tm"]["tag"] and r.get("table") == base["base"]]
    ref_lr = [r for r in jsonl(Path(base["ref_lr"]["file"])) if r.get("table") == base["base"]]
    pot = [r for r in jsonl(Path(base["potion_tm"]["file"])) if r.get("tag") == base["potion_tm"]["tag"]
           and r.get("table") == "potion-base-32M"]
    res, lines, n_pass = {}, [], 0
    for d in SETS:
        s = float(base["s"][d])
        tm = {"base": per_seed(ref, dataset=d, s=s)}
        for arm in ARMS:
            tm[arm] = per_seed(val, tag="3b-tm", arm=arm, dataset=d, s=s)
        lrv = {"base": 100.0 * float(one(ref_lr, f"base LR {d}", dataset=d)["lr_val"])}
        for arm in ARMS:
            lrv[arm] = 100.0 * float(one(lr, f"LR {arm} {d}", name=arm, dataset=d)["lr_val"])
        g = {f"{arm}-base": paired(tm[arm], tm["base"]) for arm in ARMS}
        g["3b-T-3b-L"] = paired(tm["3b-T"], tm["3b-L"])
        g["3b-T-3b-I"] = paired(tm["3b-T"], tm["3b-I"])
        glr = {f"{arm}-base": lrv[arm] - lrv["base"] for arm in ARMS}
        glr["3b-T-3b-L"] = lrv["3b-T"] - lrv["3b-L"]
        glr["3b-T-3b-I"] = lrv["3b-T"] - lrv["3b-I"]
        part = {"1_over_base": g["3b-T-base"]["mean"] >= BAR_BASE and glr["3b-T-base"] >= BAR_BASE,
                "2_over_linear": g["3b-T-3b-L"]["mean"] >= BAR_TWIN and glr["3b-T-3b-L"] >= BAR_TWIN,
                "3_over_no_teacher": g["3b-T-3b-I"]["mean"] >= BAR_TWIN and glr["3b-T-3b-I"] >= BAR_TWIN}
        meets = all(part.values())
        n_pass += meets
        ps = float(base["potion_tm"]["s"][d])
        pot_mean = st.mean(per_seed(pot, dataset=d, s=ps))
        gap = pot_mean - st.mean(tm["base"])
        res[d] = {"s": s, "tm": tm, "lr": lrv, "tm_gains": g, "lr_gains": glr, "parts": part, "meets_bar": meets,
                  "seed_sd": {k: st.stdev(v) for k, v in tm.items()}, "tm_potion": pot_mean,
                  "tm_share_of_gap_3b-T": g["3b-T-base"]["mean"] / gap if gap > 0 else None}
        lines.append(f"{d} (s {s:g}): 3b-T TM {st.mean(tm['3b-T']):.2f} vs base {st.mean(tm['base']):.2f} = "
                     f"{g['3b-T-base']['mean']:+.2f} (SE {g['3b-T-base']['se']:.2f}), LR {glr['3b-T-base']:+.2f}; "
                     f"over 3b-L TM {g['3b-T-3b-L']['mean']:+.2f} LR {glr['3b-T-3b-L']:+.2f}; "
                     f"over 3b-I TM {g['3b-T-3b-I']['mean']:+.2f} LR {glr['3b-T-3b-I']:+.2f}; "
                     f"ceiling TM {g['3b-ceil-base']['mean']:+.2f} LR {glr['3b-ceil-base']:+.2f}; parts {part}")
    verdict = "PASS" if n_pass >= 2 else "FAIL"
    why = None
    if verdict == "FAIL":
        p1 = sum(res[d]["parts"]["1_over_base"] for d in SETS)
        p12 = sum(res[d]["parts"]["1_over_base"] and res[d]["parts"]["2_over_linear"] for d in SETS)
        p13 = sum(res[d]["parts"]["1_over_base"] and res[d]["parts"]["3_over_no_teacher"] for d in SETS)
        why = ("part 1 (+0.5 over base) holds on fewer than 2 sets" if p1 < 2 else
               "the gain is a linear re-encoding of the bits (part 2)" if p12 < 2 else
               "the gain comes from the second set of concept bits, not the teacher (part 3)" if p13 < 2 else
               "parts 2 and 3 hold on different sets")
    out = {"verdict": verdict, "sets_meeting_bar": n_pass, "reason": why, "sets": res,
           "bar": "on >= 2 of 3 sets, both probes: 3b-T - base >= 0.5; 3b-T - 3b-L >= 0.3; 3b-T - 3b-I >= 0.3"}
    lines.append(f"VERDICT 3b: {verdict} ({n_pass} of 3 sets meet all three parts with both probes)" + (f"; {why}" if why else ""))
    with open(oj, "x") as f:
        f.write(json.dumps(out, indent=1) + "\n")
    with open(ot, "x") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
