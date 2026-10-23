"""Freeze a registration (design rule 1).

    python scripts/freeze.py registrations/<file>.md [extra files...]

Writes registrations/<file>.frozen.json with the UTC time read from the clock, the sha256 of
the registration, of every file under src/ and of the extra files (configs, data), and makes
the registration and the record read-only. A frozen registration is never edited; changes are
dated amendments in a new file.
"""
import json
import os
import stat
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from tally.runlog import STUDY, sha256, tree_hashes, utc  # noqa: E402

reg = os.path.abspath(sys.argv[1])
out = reg[:-3] + ".frozen.json" if reg.endswith(".md") else reg + ".frozen.json"
if os.path.exists(out):
    sys.exit(f"already frozen: {out}")
rec = {"frozen_utc": utc(), "registration": {os.path.relpath(reg, STUDY): sha256(reg)},
       "src": tree_hashes(os.path.join(STUDY, "src")),
       "extra": {os.path.relpath(os.path.abspath(p), STUDY): sha256(p) for p in sys.argv[2:]}}
with open(out, "w") as f:
    json.dump(rec, f, indent=1, sort_keys=True)
for p in (reg, out):
    os.chmod(p, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
print(json.dumps({"frozen_utc": rec["frozen_utc"], "registration_sha256": sha256(reg)}))
