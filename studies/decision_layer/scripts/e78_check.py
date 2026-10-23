#!/usr/bin/env python
"""e78 (plan logs/plan_e78_2d_counts.md): checks and the rule for 2d-a in the agent. Read only; written before any e78
number.

    e78_check.py [--mwoz-s S]     # S: the C4 agent's s* on MultiWOZ (e77), for the reported MultiWOZ line

Checks, per row (results/explore/abcd.jsonl, arm tm), stopping at the first failure:
- exactly one row per (tag, seed); every e78 row written after the plan's freeze;
- settings: dataset, encoder none, parts (view B, or view B + xcustn), lex_k 1000, clauses and T, s, seed;
- the reference seed 2 has the same train and dev bits as its seed-1 row (e74c); the two candidate seeds share their
  bits; the candidate has view B's bit count plus 1,536, and its static_info has the xcustn block of spec f8f552fd.
The rule (the bar, fixed 2026-10-19 22:06 UTC), on the mean of seeds 1 and 2, candidate minus reference:
STAR dev at least +0.5 and ABCD dev no worse than -0.5. MultiWOZ dev is reported with no rule.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
B = ["mem", "words", "xcust", "xrecent"]
BN = B + ["xcustn"]
SETS = {"star": {"clauses": 16000, "s": 16.0, "ref1": "e74c_star_Bc4_c16000_s16", "ref2": "e78_star_B_c16000_s16", "cand": "e78_star_BN_c16000_s16"},
        "abcd": {"clauses": 4000, "s": 16.0, "ref1": "e74c_abcd_Bc4_c4000_s16", "ref2": "e78_abcd_B_c4000_s16", "cand": "e78_abcd_BN_c4000_s16"}}


def need(cond: bool, msg: str) -> None:
    print(("ok   " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mwoz-s", type=float, default=None)
    args = ap.parse_args()
    frozen = json.loads((ROOT / "logs" / "plan_e78_2d_counts.frozen.json").read_text())["frozen_utc"]
    rows: dict = {}
    for line in (ROOT / "results" / "explore" / "abcd.jsonl").read_text().splitlines():
        d = json.loads(line)
        if d.get("arm") == "tm" and d.get("tag"):
            rows.setdefault((d["tag"], d["seed"]), []).append(d)

    def one(tag: str, seed: int, ds: str, parts: list, clauses: int, s: float, new: bool) -> dict:
        got = rows.get((tag, seed), [])
        need(len(got) == 1, f"{tag} seed {seed}: exactly one row")
        d = got[0]
        need(d["dataset"] == ds and d["encoder"] == "none" and list(d["parts"]) == parts and d["lex_k"] == 1000
             and d["cfg"]["n_clauses"] == clauses and d["cfg"]["T"] == clauses and d["cfg"]["s"] == s and d["cfg"]["seed"] == seed,
             f"{tag} seed {seed}: settings ({ds}, {'+'.join(parts)}, {clauses}, s {s:g})")
        if new:
            need(d["time"] > frozen, f"{tag} seed {seed}: written {d['time']}, after the freeze {frozen}")
        return d

    gaps = {}
    for ds, c in SETS.items():
        r1 = one(c["ref1"], 1, ds, B, c["clauses"], c["s"], False)
        r2 = one(c["ref2"], 2, ds, B, c["clauses"], c["s"], True)
        need(r2["bits_sha256"] == r1["bits_sha256"], f"{ds}: the reference seed 2 has seed 1's train and dev bits")
        k1 = one(c["cand"], 1, ds, BN, c["clauses"], c["s"], True)
        k2 = one(c["cand"], 2, ds, BN, c["clauses"], c["s"], True)
        need(k1["bits_sha256"] == k2["bits_sha256"], f"{ds}: the two candidate seeds share their bits")
        need(k1["n_bits"] == r1["n_bits"] + 1536, f"{ds}: candidate bits {k1['n_bits']} = view B's {r1['n_bits']} + 1,536")
        xi = k1["static_info"].get("xcustn", {})
        need(xi.get("spec") == "PLAN_text_understanding.md f8f552fd" and xi.get("n_bits") == 1536 and xi.get("count_at") == 3,
             f"{ds}: static_info has the xcustn block (count at level 3, spec f8f552fd)")
        ref = (r1["acc"] + r2["acc"]) / 2
        cand = (k1["acc"] + k2["acc"]) / 2
        gaps[ds] = 100 * (cand - ref)
        print(f"   {ds}: reference {r1['acc']:.4f} / {r2['acc']:.4f} (mean {ref:.4f}); candidate {k1['acc']:.4f} / {k2['acc']:.4f} "
              f"(mean {cand:.4f}); candidate minus reference {gaps[ds]:+.2f} points", flush=True)
    taken = gaps["star"] >= 0.5 - 1e-9 and gaps["abcd"] >= -0.5 - 1e-9
    print(f"RULE: STAR {gaps['star']:+.2f} (needs >= +0.5), ABCD {gaps['abcd']:+.2f} (needs >= -0.5) -> "
          f"{'TAKEN into the agent (then R3 and R5)' if taken else 'NOT taken'}", flush=True)
    if args.mwoz_s is not None:
        s = args.mwoz_s
        r = one(f"e77_mwoz_Bc4_c16000_s{s:g}", 1, "multiwoz", B, 16000, s, False)
        k = one(f"e78_mwoz_BN_c16000_s{s:g}", 1, "multiwoz", BN, 16000, s, True)
        print(f"   multiwoz (reported, no rule): view B {r['acc']:.4f}, view B + xcustn {k['acc']:.4f}: {100 * (k['acc'] - r['acc']):+.2f} points (seed 1)")


if __name__ == "__main__":
    main()
