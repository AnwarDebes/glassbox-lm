#!/usr/bin/env python
"""Run abcd_explore.py or abcd_confirm.py with the C4 table and an additive text-stack block (added 2026-10-19).

    stack_run.py --table c4words-S1000-v1 scripts/abcd_explore.py --dataset star --encoder none \
        --parts mem,words,xcust,xrecent,xcustn --arms tm --clauses 16000 --s 16 --seed 1 --tag ...

table_run.py's way, plus one block: dl.count_text.patch (the count table; potion's table cannot load), then
dl.count_bits.patch (dl.abcd.ABCDView is dl.count_bits.ABCDViewN, which accepts the part "xcustn", 2d-a), then the
unedited target runs (runpy, as __main__). The frozen files (table_run.py, abcd_explore.py, abcd_confirm.py,
src/dl/abcd.py, static_text.py, count_text.py) are not edited. Protocol: logs/protocol_text_stack_agent_checks.md.
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

from dl import count_bits, count_text  # noqa: E402

TARGETS = ("abcd_explore.py", "abcd_confirm.py")
PARTS_OK = {"mem", "words", "xcust", "xrecent", "cwords", "xcrecent", count_bits.PART}


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
        raise SystemExit("stack_run.py needs --encoder none (no encoder vector is loaded)")
    parts = (arg(rest, "--parts") or "").split(",")
    if not parts or not set(parts) <= PARTS_OK:
        raise SystemExit(f"--parts must be a subset of {sorted(PARTS_OK)}, got {parts}")
    meta = count_text.patch(name)
    count_bits.patch()
    print(json.dumps({"stack_run": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "target": str(target.resolve().relative_to(ROOT)),
                      "target_sha256": hashlib.sha256(target.resolve().read_bytes()).hexdigest(),
                      "count_bits_sha256": hashlib.sha256((ROOT / "src" / "dl" / "count_bits.py").read_bytes()).hexdigest(),
                      "argv": rest, "table": meta}), flush=True)
    sys.argv = [str(target)] + rest
    runpy.run_path(str(target.resolve()), run_name="__main__")


if __name__ == "__main__":
    main()
