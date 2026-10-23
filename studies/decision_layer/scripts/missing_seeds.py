#!/usr/bin/env python
"""List the registered router seeds that have no result file (for example after a CUDA OOM crash)
and write job lines to re-run them.

    missing_seeds.py R4  -> logs/jobs_R4_missing.txt
    missing_seeds.py R4b -> logs/jobs_R4b_missing.txt
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
reg = sys.argv[1]
cells = ["clinc150_minilm", "banking77_minilm", "hwu64_minilm", "clinc150_bge-base", "banking77_bge-base", "hwu64_bge-base"]
extra = {"R4": "--epochs 80 --readout", "R4b": "--clauses 16000"}[reg]
lines = []
for cell in cells:
    ds, enc = cell.split("_", 1)
    have = {int(p.stem[4:]) for p in (ROOT / "results" / "confirm" / reg / cell).glob("seed*.json")}
    miss = [k for k in range(1, 11) if k not in have]
    print(cell, "missing", miss)
    if miss:
        lines.append(f".venv-dl/bin/python scripts/confirm_router.py --registration {reg} {extra} --dataset {ds} --encoder {enc} --name {cell} --seeds {' '.join(map(str, miss))}")
(ROOT / "logs" / f"jobs_{reg}_missing.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
