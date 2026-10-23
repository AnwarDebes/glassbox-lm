#!/usr/bin/env python
"""R22: copy the registered reference arms into R22's folder, so scripts/report_abcd.py can compare them with the tuned
arms in one report. Copies seed<k>.json, seed<k>.npz and bit_names.json (never the models or LR weights); nothing is
recomputed. Prints the SHA-256 of every copied file (for the registration).

- R22star: R21star's tm_mem_c4 (s 8) -> tm_mem_c4_s8_R21; R21star's lr_mem_c4 (wide grid) -> lr_mem_c4_R21;
  R18's tm_mem (bge-large, s 8) -> tm_mem_bge_s8_R18
Refuses to overwrite. Added 2026-10-19 for R22.
"""

import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONF = ROOT / "results" / "confirm"
PLAN = {"R22star": [("R21star", "tm_mem_c4", "tm_mem_c4_s8_R21"), ("R21star", "lr_mem_c4", "lr_mem_c4_R21"),
                    ("R18", "tm_mem", "tm_mem_bge_s8_R18")]}


def main() -> None:
    for dst_reg, arms in PLAN.items():
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
