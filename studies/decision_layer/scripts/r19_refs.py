#!/usr/bin/env python
"""R19: copy the bge-large reference arms into R19's folders, so scripts/report_abcd.py can compare them with
the no-network arms in one report. Copies seed<k>.json, seed<k>.npz and bit_names.json (never the models);
nothing is recomputed. Prints the SHA-256 of every copied file (for the registration).

- R19star: R18's tm_mem, tm_nomem, lr_mem -> tm_mem_R18, tm_nomem_R18, lr_mem_R18
- R19abcd: R9b's tm_mem_short, tm_nomem_short, lr_mem_short -> tm_mem_R9b, tm_nomem_R9b, lr_mem_R9b
Refuses to overwrite. Added 2026-10-17 for R19.
"""

import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONF = ROOT / "results" / "confirm"
PLAN = {"R19star": ("R18", {"tm_mem": "tm_mem_R18", "tm_nomem": "tm_nomem_R18", "lr_mem": "lr_mem_R18"}),
        "R19abcd": ("R9b", {"tm_mem_short": "tm_mem_R9b", "tm_nomem_short": "tm_nomem_R9b", "lr_mem_short": "lr_mem_R9b"})}


def main() -> None:
    for dst_reg, (src_reg, arms) in PLAN.items():
        for src, dst in arms.items():
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
