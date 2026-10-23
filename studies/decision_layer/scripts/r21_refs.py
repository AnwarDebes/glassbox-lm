#!/usr/bin/env python
"""R21: copy the registered reference arms into R21's folders, so scripts/report_abcd.py can compare them with the
C4-table arms in one report. Copies seed<k>.json, seed<k>.npz and bit_names.json (never the models or LR weights);
nothing is recomputed. Prints the SHA-256 of every copied file (for the registration).

- R21star: R19star's tm_mem_static, tm_nomem_static, lr_mem_static -> tm_mem_static_R19, tm_nomem_static_R19,
  lr_mem_static_R19 (the potion agent); R18's tm_mem, tm_nomem, lr_mem -> tm_mem_R18, tm_nomem_R18, lr_mem_R18
  (bge-large)
- R21abcd: R19abcd's three arms in the same way; R9b's tm_mem_short, tm_nomem_short, lr_mem_short -> tm_mem_R9b,
  tm_nomem_R9b, lr_mem_R9b (bge-large, 10 seeds)

    r21_refs.py R21star R21abcd      # only the named registrations' folders
Refuses to overwrite. Added 2026-10-19 for R21 (the agent on the C4-word table).
"""

import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONF = ROOT / "results" / "confirm"
PLAN = {"R21star": [("R19star", "tm_mem_static", "tm_mem_static_R19"), ("R19star", "tm_nomem_static", "tm_nomem_static_R19"),
                    ("R19star", "lr_mem_static", "lr_mem_static_R19"), ("R18", "tm_mem", "tm_mem_R18"),
                    ("R18", "tm_nomem", "tm_nomem_R18"), ("R18", "lr_mem", "lr_mem_R18")],
        "R21abcd": [("R19abcd", "tm_mem_static", "tm_mem_static_R19"), ("R19abcd", "tm_nomem_static", "tm_nomem_static_R19"),
                    ("R19abcd", "lr_mem_static", "lr_mem_static_R19"), ("R9b", "tm_mem_short", "tm_mem_R9b"),
                    ("R9b", "tm_nomem_short", "tm_nomem_R9b"), ("R9b", "lr_mem_short", "lr_mem_R9b")]}


def main() -> None:
    regs = sys.argv[1:]
    if not regs or any(r not in PLAN for r in regs):
        raise SystemExit(f"usage: r21_refs.py {' '.join(PLAN)}")
    for dst_reg in regs:
        for src_reg, src, dst in PLAN[dst_reg]:
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
