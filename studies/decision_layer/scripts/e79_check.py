#!/usr/bin/env python
"""e79 (plan logs/plan_e79_c4_prototypes.md, frozen 2026-10-20T19:00:04Z): checks and the rule for view A+B on the C4
table, in the plan's "as written" branch (e78 did not take 2d-a). Read only; written before any e79 number.

    e79_check.py

Checks, stopping at the first failure: one row per (tag, seed); e79 rows written after the freeze; settings (dataset,
encoder, parts, lex_k 1000, clauses and T, s, seed); the reference seed 2 shares seed 1's train and dev bits; the two
candidate seeds share their bits; the candidate has more bits than view B and its static_info has view B's xcust and
xrecent blocks on the C4 table file.
The rule, on the mean of seeds 1 and 2, candidate minus reference: MultiWOZ dev at least +1.0, STAR dev no worse than
-0.5, ABCD dev no worse than -0.5.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
B = ["mem", "words", "xcust", "xrecent"]
AB = ["mem", "intent", "step", "words", "xcust", "xrecent"]
TABLE_FILE = "e1771d9eaf4a07f71c532fe4f3ef91909334cbdcb4d7785ff0bb7ed0e51c5c0a"
SETS = {"multiwoz": {"clauses": 16000, "s": 8.0, "ref": [("e77_mwoz_Bc4_c16000_s8", 1, False), ("e79_mwoz_B_c16000_s8", 2, True)],
                     "cand": "e79_mwoz_AB_c16000_s8", "bar": 1.0},
        "star": {"clauses": 16000, "s": 16.0, "ref": [("e74c_star_Bc4_c16000_s16", 1, False), ("e78_star_B_c16000_s16", 2, False)],
                 "cand": "e79_star_AB_c16000_s16", "bar": -0.5},
        "abcd": {"clauses": 4000, "s": 16.0, "ref": [("e74c_abcd_Bc4_c4000_s16", 1, False), ("e78_abcd_B_c4000_s16", 2, False)],
                 "cand": "e79_abcd_AB_c4000_s16", "bar": -0.5}}


def need(cond: bool, msg: str) -> None:
    print(("ok   " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        sys.exit(1)


def main() -> None:
    frozen = json.loads((ROOT / "logs" / "plan_e79_c4_prototypes.frozen.json").read_text())["frozen_utc"]
    rows: dict = {}
    for line in (ROOT / "results" / "explore" / "abcd.jsonl").read_text().splitlines():
        d = json.loads(line)
        if d.get("arm") == "tm" and d.get("tag"):
            rows.setdefault((d["tag"], d["seed"]), []).append(d)

    def one(tag: str, seed: int, ds: str, enc: str, parts: list, clauses: int, s: float, new: bool) -> dict:
        got = rows.get((tag, seed), [])
        need(len(got) == 1, f"{tag} seed {seed}: exactly one row")
        d = got[0]
        need(d["dataset"] == ds and d["encoder"] == enc and list(d["parts"]) == parts and d["lex_k"] == 1000
             and d["cfg"]["n_clauses"] == clauses and d["cfg"]["T"] == clauses and d["cfg"]["s"] == s and d["cfg"]["seed"] == seed,
             f"{tag} seed {seed}: settings ({ds}, {enc}, {'+'.join(parts)}, {clauses}, s {s:g})")
        if new:
            need(d["time"] > frozen, f"{tag} seed {seed}: written {d['time']}, after the freeze {frozen}")
        return d

    gaps, ok_all = {}, True
    for ds, c in SETS.items():
        refs = [one(t, sd, ds, "none", B, c["clauses"], c["s"], new) for t, sd, new in c["ref"]]
        need(refs[0]["bits_sha256"] == refs[1]["bits_sha256"], f"{ds}: the reference seeds share their train and dev bits")
        cands = [one(c["cand"], sd, ds, "c4mean", AB, c["clauses"], c["s"], True) for sd in (1, 2)]
        need(cands[0]["bits_sha256"] == cands[1]["bits_sha256"], f"{ds}: the two candidate seeds share their bits")
        need(cands[0]["n_bits"] > refs[0]["n_bits"], f"{ds}: candidate bits {cands[0]['n_bits']} > view B's {refs[0]['n_bits']}")
        si = cands[0]["static_info"]
        need(all(si[k]["table_file_sha256"] == TABLE_FILE for k in ("xcust", "xrecent")), f"{ds}: xcust and xrecent on the C4 table file")
        ref = sum(r["acc"] for r in refs) / 2
        cand = sum(k["acc"] for k in cands) / 2
        gaps[ds] = 100 * (cand - ref)
        ok = gaps[ds] >= c["bar"] - 1e-9
        ok_all &= ok
        print(f"   {ds}: view B {refs[0]['acc']:.4f} / {refs[1]['acc']:.4f} (mean {ref:.4f}); A+B {cands[0]['acc']:.4f} / {cands[1]['acc']:.4f} "
              f"(mean {cand:.4f}); A+B minus B {gaps[ds]:+.2f} (needs >= {c['bar']:+.1f}): {'holds' if ok else 'fails'}", flush=True)
    print(f"RULE: {'TAKEN (then R3, R5, and the R24 gate again)' if ok_all else 'NOT taken'}", flush=True)


if __name__ == "__main__":
    main()
