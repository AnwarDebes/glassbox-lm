#!/usr/bin/env python
"""R25: copy the registered reference arms into R25's folders, so scripts/report_abcd.py can compare them with the
refined-table arms in one report. Copies seed<k>.json, seed<k>.npz and bit_names.json (never the models); nothing is
recomputed. Prints the SHA-256 of every copied file (for the registration).

- R25star: R22star's tm_mem_c4_s16 -> tm_mem_c4_s16_R22 and tm_mem_bge_s16 -> tm_mem_bge_s16_R22 (5 seeds each);
- R25abcd: R23abcd's tm_mem_c4_s16 -> tm_mem_c4_s16_R23 and tm_mem_bge_s24 -> tm_mem_bge_s24_R23 (5 seeds each).
Only the registration folders of the datasets whose go-bars pass are written (--datasets). Plan: registrations/R25_refined_table_agent.md.

Refuses to overwrite. Added 2026-10-21 for R25 (adapted from r23_refs.py).
"""

import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONF = ROOT / "results" / "confirm"
PLAN = {"R25star": [("R22star", "tm_mem_c4_s16", "tm_mem_c4_s16_R22"), ("R22star", "tm_mem_bge_s16", "tm_mem_bge_s16_R22")],
        "R25abcd": [("R23abcd", "tm_mem_c4_s16", "tm_mem_c4_s16_R23"), ("R23abcd", "tm_mem_bge_s24", "tm_mem_bge_s24_R23")]}


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True, choices=["star", "abcd"])
    want = {"star": "R25star", "abcd": "R25abcd"}
    args = ap.parse_args()
    for dst_reg, arms in PLAN.items():
        if dst_reg not in {want[d] for d in args.datasets}:
            continue
        for src_reg, src, dst in arms:
            sd, dd = CONF / src_reg / src, CONF / dst_reg / dst
            if dd.exists():
                raise SystemExit(f"exists: {dd}")
            dd.mkdir(parents=True)
            files = sorted(p for p in sd.iterdir() if (p.name.startswith("seed") and p.suffix in (".json", ".npz")) or p.name == "bit_names.json")
            for p in files:
                shutil.copy2(p, dd / p.name)
                print(f"{dst_reg}/{dst}/{p.name} {hashlib.sha256(p.read_bytes()).hexdigest()} (from {src_reg}/{src})", flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)
