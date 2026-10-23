#!/usr/bin/env python
"""Run abcd_explore.py or abcd_confirm.py with the existsT parts on a count table (added 2026-10-19 for e72).

    table_run.py --table c4words-S1000-v1 scripts/abcd_explore.py --dataset abcd --encoder none \
        --parts mem,words,xcust,xrecent --arms tm --clauses 4000 --seed 1 --save --tag e72_abcd_Bc4_c4000

Plan: logs/plan_e72_count_table.md. The run scripts and src/dl/static_text.py, dl/abcd.py are not edited (their
hashes are in the e70 and e71 freezes). Before the unedited target runs (runpy, as __main__):
- dl.count_text.patch: dl.static_text.ExistsT builds ExistsTC on the named table, so ABCDView's xcust, xrecent and
  xcrecent read it with its tokenizer; dl.static_text.table and StaticTable raise, so potion's table cannot load;
- the target must get --encoder none (data/emb/<ds>__none.npz from scripts/none_prepare.py: empty arrays) and parts
  that read no encoder vector, so no vector of any encoder is loaded.
The target's rows carry the table in static_info (name, file and array SHA-256, tokenizer). This script prints the
target's SHA-256 and the table's meta first.
"""

from __future__ import annotations

import hashlib
import json
import runpy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dl import count_text  # noqa: E402

TARGETS = ("abcd_explore.py", "abcd_confirm.py")
NO_ENCODER_PARTS = {"mem", "words", "xcust", "xrecent", "cwords", "xcrecent"}


def arg(rest: list[str], flag: str) -> str | None:
    return rest[rest.index(flag) + 1] if flag in rest and rest.index(flag) + 1 < len(rest) else None


def main() -> None:
    a = sys.argv[1:]
    if len(a) < 3 or a[0] != "--table" or a[1] not in count_text.TABLES:
        raise SystemExit(__doc__ + f"\nknown tables: {sorted(count_text.TABLES)}")
    name, target, rest = a[1], Path(a[2]), a[3:]
    if target.name not in TARGETS or target.resolve().parent != ROOT / "scripts":
        raise SystemExit(f"target must be one of scripts/{TARGETS}")
    if arg(rest, "--encoder") != "none":
        raise SystemExit("table_run.py needs --encoder none (no encoder vector is loaded)")
    parts = (arg(rest, "--parts") or "").split(",")
    if not parts or not set(parts) <= NO_ENCODER_PARTS:
        raise SystemExit(f"--parts must be a subset of {sorted(NO_ENCODER_PARTS)}, got {parts}")
    meta = count_text.patch(name)
    print(json.dumps({"table_run": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "target": str(target.resolve().relative_to(ROOT)),
                      "target_sha256": hashlib.sha256(target.resolve().read_bytes()).hexdigest(), "argv": rest, "table": meta}), flush=True)
    sys.argv = [str(target)] + rest
    runpy.run_path(str(target.resolve()), run_name="__main__")


if __name__ == "__main__":
    main()
