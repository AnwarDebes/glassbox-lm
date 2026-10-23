#!/usr/bin/env python
"""Freeze a registration: write <name>.frozen.json with the UTC time and the SHA-256 of the
registration text and of every source file of the study (src/, scripts/). Refuses to overwrite."""
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
reg = Path(sys.argv[1]).resolve()
out = reg.with_suffix(".frozen.json")
if out.exists():
    raise SystemExit(f"already frozen: {out}")
h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
rec = {
    "registration": str(reg.relative_to(ROOT)),
    "sha256": h(reg),
    "frozen_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "sources": {str(p.relative_to(ROOT)): h(p) for d in ("src", "scripts") for p in sorted((ROOT / d).rglob("*.py")) + sorted((ROOT / d).rglob("*.sh"))},
}
out.write_text(json.dumps(rec, indent=1))
print(rec["frozen_utc"], rec["sha256"][:16], len(rec["sources"]), "source files")
