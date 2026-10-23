#!/usr/bin/env python
"""List source files whose SHA-256 differs from a registration's frozen record (or are new)."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
rec = json.loads(Path(sys.argv[1]).read_text())
h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
now = {str(p.relative_to(ROOT)): h(p) for d in ("src", "scripts") for p in sorted((ROOT / d).rglob("*.py")) + sorted((ROOT / d).rglob("*.sh"))}
for f in sorted(set(now) | set(rec["sources"])):
    a, b = rec["sources"].get(f), now.get(f)
    if a != b:
        print(("NEW " if a is None else "GONE " if b is None else "CHANGED ") + f)
