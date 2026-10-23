#!/usr/bin/env python
"""Run abcd_explore.py or abcd_confirm.py with view A+B on the C4 table (e79; added 2026-10-20).

    mean_run.py --table c4words-S1000-v1 scripts/abcd_explore.py --dataset multiwoz --encoder c4mean \
        --parts mem,intent,step,words,xcust,xrecent --arms tm --clauses 16000 --s 8 --seed 1 --save --tag ...

Plan: logs/plan_e79_c4_prototypes.md. table_run.py's way: dl.count_text.patch (xcust and xrecent read the C4 table;
potion's table cannot load), dl.count_bits.patch (so "xcustn" is available for the plan's stacking line), then the
unedited target runs (runpy, as __main__). Intent and step read data/emb/<ds>__c4mean.npz from
scripts/c4mean_prepare.py: C4 mean vectors, no neural network. The cache's provenance must name this table's file. The
frozen files (table_run.py, stack_run.py, abcd_explore.py, abcd_confirm.py, src/dl/*) are not edited.
"""

from __future__ import annotations

import hashlib
import json
import runpy
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dl import count_bits, count_text  # noqa: E402

TARGETS = ("abcd_explore.py", "abcd_confirm.py")
PARTS_OK = {"mem", "intent", "step", "words", "xcust", "xrecent", count_bits.PART}


def arg(rest: list[str], flag: str) -> str | None:
    return rest[rest.index(flag) + 1] if flag in rest and rest.index(flag) + 1 < len(rest) else None


def main() -> None:
    a = sys.argv[1:]
    if len(a) < 3 or a[0] != "--table" or a[1] not in count_text.TABLES:
        raise SystemExit(__doc__ + f"\nknown tables: {sorted(count_text.TABLES)}")
    name, target, rest = a[1], Path(a[2]), a[3:]
    if target.name not in TARGETS or target.resolve().parent != ROOT / "scripts":
        raise SystemExit(f"target must be one of scripts/{TARGETS}")
    if arg(rest, "--encoder") != "c4mean":
        raise SystemExit("mean_run.py needs --encoder c4mean (the C4 mean-vector cache)")
    parts = (arg(rest, "--parts") or "").split(",")
    if not parts or not set(parts) <= PARTS_OK:
        raise SystemExit(f"--parts must be a subset of {sorted(PARTS_OK)}, got {parts}")
    ds = arg(rest, "--dataset") or "abcd"
    cache = ROOT / "data" / "emb" / f"{ds}__c4mean.npz"
    prov = json.loads(str(np.load(cache)["provenance"]))
    if prov["table"]["table_file_sha256"] != count_text.TABLES[name][1] or prov["dataset"] != ds:
        raise SystemExit(f"{cache}: not the C4 mean cache of table {name} for {ds}")
    meta = count_text.patch(name)
    count_bits.patch()
    print(json.dumps({"mean_run": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "target": str(target.resolve().relative_to(ROOT)),
                      "target_sha256": hashlib.sha256(target.resolve().read_bytes()).hexdigest(),
                      "cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(), "argv": rest, "table": meta}), flush=True)
    sys.argv = [str(target)] + rest
    runpy.run_path(str(target.resolve()), run_name="__main__")


if __name__ == "__main__":
    main()
