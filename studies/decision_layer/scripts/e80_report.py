#!/usr/bin/env python
"""e80's readings beside its rule (plan logs/plan_e80_2d_recent.md, frozen 2026-10-20T20:53:40Z; dated note
logs/plan_e80_note_2026-10-25.md). Read only, CPU only, no rule: the rule is scripts/e80_check.py's, unchanged.
Written 2026-10-25, before any e80 number. No test file is read.

    e80_report.py > logs/e80_report.out      # after logs/jobs_e80.txt and scripts/e80_check.py

Per dataset (STAR and ABCD): the reference (view B: e74c's row for seed 1, e78's for seed 2) and the candidate (view
B + xcrecent, seeds 1 and 2), each read three ways from its row: the best-dev snapshot (the row's acc, the statistic
of the rule), the final epoch (the last entry of its history), and the mean of its last 10 evaluations. For each:
the mean of the two seeds, and candidate minus reference in points. MultiWOZ: seed 1 only, view B (e77's row)
against view B + xcrecent (the frozen checker's MultiWOZ line calls the candidate "view B + xcustn", a leftover of its
adaptation from e78_check.py).
Caveats printed with the readings: selection is optimistic (the best-dev snapshot is the best of 20 dev evaluations
on the dev split the rule reads); no multiplicity correction (one rule with two conditions; MultiWOZ is a reading; no
superiority claim).
Refuses without the note's freeze record (its text must hash to it), unless every (tag, seed) has exactly one tm row,
and unless every candidate row was written after the note's freeze.
Tested by tests/test_e80_report.py (made-up rows).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTE = "plan_e80_note_2026-10-25"
SETS = {"star": ("e74c_star_Bc4_c16000_s16", "e78_star_B_c16000_s16", "e80_star_BX_c16000_s16"),
        "abcd": ("e74c_abcd_Bc4_c4000_s16", "e78_abcd_B_c4000_s16", "e80_abcd_BX_c4000_s16")}
MWOZ = ("e77_mwoz_Bc4_c16000_s8", "e80_mwoz_BX_c16000_s8")
WAYS = (("best-dev", lambda d: d["acc"]), ("final epoch", lambda d: d["history"][-1]["val_acc"]),
        ("last-10 mean", lambda d: sum(e["val_acc"] for e in d["history"][-10:]) / len(d["history"][-10:])))


def refuse(msg: str) -> None:
    raise SystemExit(f"e80_report: {msg}")


def main() -> None:
    rec, txt = ROOT / "logs" / f"{NOTE}.frozen.json", ROOT / "logs" / f"{NOTE}.md"
    if not rec.exists():
        refuse(f"{rec.name} is missing")
    r = json.loads(rec.read_text())
    if not txt.exists() or hashlib.sha256(txt.read_bytes()).hexdigest() != r["sha256"]:
        refuse(f"logs/{NOTE}.md does not hash to its freeze record")
    rows: dict = {}
    for line in (ROOT / "results" / "explore" / "abcd.jsonl").read_text().splitlines():
        d = json.loads(line)
        if d.get("arm") == "tm" and d.get("tag"):
            rows.setdefault((d["tag"], d["seed"]), []).append(d)

    def one(tag: str, seed: int, cand: bool) -> dict:
        got = rows.get((tag, seed), [])
        if len(got) != 1:
            refuse(f"{tag} seed {seed}: {len(got)} rows (need exactly 1)")
        if cand and got[0]["time"] <= r["frozen_utc"]:
            refuse(f"{tag} seed {seed}: written before the note's freeze")
        return got[0]

    print("e80 readings (dev only; no rule: the rule is scripts/e80_check.py's); each value is the mean of seeds 1 and 2", flush=True)
    for ds, (ref1, ref2, cand) in SETS.items():
        refs, cands = (one(ref1, 1, False), one(ref2, 2, False)), (one(cand, 1, True), one(cand, 2, True))
        parts = []
        for name, f in WAYS:
            a, b = sum(map(f, refs)) / 2, sum(map(f, cands)) / 2
            parts.append(f"{name}: view B {a:.4f}, view B + xcrecent {b:.4f}, {100 * (b - a):+.2f} points")
        print(f"  {ds}: " + "; ".join(parts) + (" (the best-dev line is the rule's statistic)" if ds == "star" else ""))
    r0, c0 = one(MWOZ[0], 1, False), one(MWOZ[1], 1, True)
    print("  multiwoz (seed 1 only, a reading with no rule): " + "; ".join(
        f"{name}: view B {f(r0):.4f}, view B + xcrecent {f(c0):.4f}, {100 * (f(c0) - f(r0)):+.2f} points" for name, f in WAYS))
    print("caveats: selection is optimistic (the best-dev snapshot is the best of 20 dev evaluations on the dev split the rule reads);"
          " no multiplicity correction: one rule with two conditions, MultiWOZ is a reading, no superiority claim")


if __name__ == "__main__":
    main()
