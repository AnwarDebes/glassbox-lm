#!/usr/bin/env python
"""e72e (note logs/plan_e72e_guards_c4_s16.md): e72d's checks and guards on R23's five tuned ABCD models.

    verify_policy_c4_e72e.py --check encoding --threads 1 --tag e72e
    verify_policy_c4_e72e.py --check properties [--guard] --threads 1 --tag e72e
    verify_policy_c4_e72e.py --check guardcost --tag e72e

A thin wrapper: it points scripts/verify_policy_c4.py (unedited) at results/confirm/R23abcd/tm_mem_c4_s16 (view B on the
C4 table at s 16; the models of R23's C4 arm) and writes to results/explore/verify_e72e.jsonl. Everything else is e72d's
and, through it, e70's: the 50 checks, the exact MILP encoding and HiGHS options, the level-B domain rows from the
models' bit names, and the guards with M recomputed per model. Property runs start only after this note is frozen.
Dev and proof work only. Added 2026-10-19.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(1, str(ROOT / "scripts"))

import verify_policy_c4 as vpc  # noqa: E402

vpc.SRC = ROOT / "results" / "confirm" / "R23abcd" / "tm_mem_c4_s16"
vpc.OUT = ROOT / "results" / "explore" / "verify_e72e.jsonl"

if __name__ == "__main__":
    if "properties" in sys.argv and not (ROOT / "logs" / "plan_e72e_guards_c4_s16.frozen.json").exists():
        raise SystemExit("e72e property runs start only after logs/plan_e72e_guards_c4_s16.md is frozen")
    if "--tag" not in sys.argv:
        raise SystemExit("pass --tag e72e")
    vpc.main()
