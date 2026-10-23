#!/usr/bin/env python
"""e76 report (read only; plan logs/plan_e76_wordedit_c4.md): e71's per-group and per-state tables for the agent with no
neural network at any stage (R23abcd tm_mem_c4_s16), from results/explore/verify_e76.jsonl.

    report_e76.py --tag e76 --seed 1 > logs/report_e76_seed1.out

scripts/report_e71.py is imported unedited; only its input file is set to e76's (no k = 3/5 rows exist for e76).
Added 2026-10-19.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import report_e71  # noqa: E402

report_e71.OUT = ROOT / "results" / "explore" / "verify_e76.jsonl"

if __name__ == "__main__":
    if "--k35-tag" not in sys.argv:
        sys.argv += ["--k35-tag", "none"]
    report_e71.main()
