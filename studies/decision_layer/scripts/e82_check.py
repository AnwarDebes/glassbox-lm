#!/usr/bin/env python
"""e82 (plan logs/plan_e82_refined_table_agent.md): checks and the R4 rule for a new count table in the agent. Read
only; written before any e82 number.

    e82_check.py --name <table name> --table-sha <table.npz SHA-256>

Rows: results/explore/abcd.jsonl (arm tm). The candidate tags are e82_<ds>_Bt_c<clauses>_s<s> (view B on the new
table, seeds 1 and 2). The references are view B on the C4 table at the same settings (the plan's table). Checks,
stopping at the first failure:
- one row per (tag, seed), candidates written after the plan's freeze;
- the settings: dataset, encoder none, view B's parts, lex_k 1000, clauses and T, s, seed;
- the references' seeds share their bits, and so do the candidate's;
- the candidate's xcust and xrecent were built on the named table (its file SHA-256), and its bit count is reported.
The rule (the map's R4, fixed in the plan): the new table is taken into the agent only if no dataset loses more than 0.5
points (candidate minus reference, mean of seeds 1 and 2). Every difference is reported.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
B = ["mem", "words", "xcust", "xrecent"]
SETS = {"star": {"clauses": 16000, "s": 16.0, "ref": [("e74c_star_Bc4_c16000_s16", 1), ("e78_star_B_c16000_s16", 2)]},
        "abcd": {"clauses": 4000, "s": 16.0, "ref": [("e74c_abcd_Bc4_c4000_s16", 1), ("e78_abcd_B_c4000_s16", 2)]},
        "multiwoz": {"clauses": 16000, "s": 8.0, "ref": [("e77_mwoz_Bc4_c16000_s8", 1), ("e79_mwoz_B_c16000_s8", 2)]}}


def need(cond: bool, msg: str) -> None:
    print(("ok   " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--table-sha", required=True)
    args = ap.parse_args()
    frozen = json.loads((ROOT / "logs" / "plan_e82_refined_table_agent.frozen.json").read_text())["frozen_utc"]
    rows: dict = {}
    for line in (ROOT / "results" / "explore" / "abcd.jsonl").read_text().splitlines():
        d = json.loads(line)
        if d.get("arm") == "tm" and d.get("tag"):
            rows.setdefault((d["tag"], d["seed"]), []).append(d)

    def one(tag: str, seed: int, ds: str, c: dict, new: bool) -> dict:
        got = rows.get((tag, seed), [])
        need(len(got) == 1, f"{tag} seed {seed}: exactly one row")
        d = got[0]
        need(d["dataset"] == ds and d["encoder"] == "none" and list(d["parts"]) == B and d["lex_k"] == 1000
             and d["cfg"]["n_clauses"] == c["clauses"] and d["cfg"]["T"] == c["clauses"] and d["cfg"]["s"] == c["s"] and d["cfg"]["seed"] == seed,
             f"{tag} seed {seed}: settings ({ds}, view B, {c['clauses']}, s {c['s']:g})")
        if new:
            need(d["time"] > frozen, f"{tag} seed {seed}: written {d['time']}, after the freeze {frozen}")
        return d

    gaps = {}
    for ds, c in SETS.items():
        refs = [one(t, sd, ds, c, False) for t, sd in c["ref"]]
        need(refs[0]["bits_sha256"] == refs[1]["bits_sha256"], f"{ds}: the reference seeds share their bits")
        tag = f"e82_{ds}_Bt_c{c['clauses']}_s{c['s']:g}"
        cands = [one(tag, sd, ds, c, True) for sd in (1, 2)]
        need(cands[0]["bits_sha256"] == cands[1]["bits_sha256"], f"{ds}: the candidate seeds share their bits")
        si = cands[0]["static_info"]
        need(all(si[k]["table"] == args.name and si[k]["table_file_sha256"] == args.table_sha for k in ("xcust", "xrecent")),
             f"{ds}: xcust and xrecent on table {args.name} ({args.table_sha[:16]}); candidate {cands[0]['n_bits']} bits, reference {refs[0]['n_bits']}")
        ref = sum(r["acc"] for r in refs) / 2
        cand = sum(k["acc"] for k in cands) / 2
        gaps[ds] = 100 * (cand - ref)
        print(f"   {ds}: C4 table {refs[0]['acc']:.4f} / {refs[1]['acc']:.4f} (mean {ref:.4f}); {args.name} {cands[0]['acc']:.4f} / "
              f"{cands[1]['acc']:.4f} (mean {cand:.4f}); difference {gaps[ds]:+.2f}", flush=True)
    taken = all(g >= -0.5 - 1e-9 for g in gaps.values())
    print(f"RULE (R4: no dataset below -0.5): {', '.join(f'{k} {v:+.2f}' for k, v in gaps.items())} -> "
          f"{'TAKEN into the agent (then R3, R5, the R24 gate)' if taken else 'NOT taken'}", flush=True)


if __name__ == "__main__":
    main()
