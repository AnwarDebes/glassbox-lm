#!/usr/bin/env python
"""R20: copy the registered reference arms into R20's folders, so scripts/report_abcd.py can compare them with
the closed-loop and customer-only arms in one report. Copies seed<k>.json, seed<k>.npz and bit_names.json (never
the models or LR weights); nothing is recomputed. Prints the SHA-256 of every copied file (for the registration).

- R20star: R19star's tm_mem_static, tm_nomem_static, lr_mem_static -> tm_mem_static_R19, tm_nomem_static_R19,
  lr_mem_static_R19 (open loop); R18's tm_mem -> tm_mem_R18 (open loop)
- R20abcd: R19abcd's tm_mem_static, tm_nomem_static -> tm_mem_static_R19, tm_nomem_static_R19 (open loop);
  R13's tm_mem_short_closed -> tm_mem_closed_R13 (bge-large, closed loop, R9b's 10 models)
Refuses to overwrite. Added 2026-10-18 for R20.
"""

import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONF = ROOT / "results" / "confirm"
PLAN = {"R20star": [("R19star", "tm_mem_static", "tm_mem_static_R19"), ("R19star", "tm_nomem_static", "tm_nomem_static_R19"),
                    ("R19star", "lr_mem_static", "lr_mem_static_R19"), ("R18", "tm_mem", "tm_mem_R18")],
        "R20abcd": [("R19abcd", "tm_mem_static", "tm_mem_static_R19"), ("R19abcd", "tm_nomem_static", "tm_nomem_static_R19"),
                    ("R13", "tm_mem_short_closed", "tm_mem_closed_R13")]}


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
