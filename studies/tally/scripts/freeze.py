"""Freeze a registration (design rule 1).

    python scripts/freeze.py registrations/<file>.md [extra files...]

Writes registrations/<file>.frozen.json with the UTC time read from the clock, the sha256 of
the registration, of every file under src/ and of the extra files (configs, data), and makes
the registration and the record read-only. A frozen registration is never edited; changes are
dated amendments in a new file.

What is pinned: the registration, every file under src/ (the extensions that runlog.tree_hashes
lists: .py, .sh, .md, .json, .toml) and the extra files named on the command line. Nothing under
scripts/, tests/ or data/ is pinned unless it is named as an extra file, so a registration whose
job runs a script (for example scripts/a2_run.py or scripts/b1_launch.sh) or reads data files
names them here. The read-only mode is not part of the record: a git checkout or a copy resets
it, and the hashes in the record are what shows that a file is unchanged.
"""
import json
import os
import stat
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from tally.runlog import STUDY, sha256, tree_hashes, utc  # noqa: E402

USAGE = "usage: python scripts/freeze.py registrations/<file>.md [extra files...]"


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE, file=sys.stderr if not argv else sys.stdout)
        return 2 if not argv else 0
    missing = [p for p in argv if not os.path.isfile(p)]
    if missing:
        print(f"not a file: {', '.join(missing)}\n{USAGE}", file=sys.stderr)
        return 2
    reg = os.path.abspath(argv[0])
    out = reg[:-3] + ".frozen.json" if reg.endswith(".md") else reg + ".frozen.json"
    if os.path.exists(out):
        print(f"already frozen: {out}", file=sys.stderr)
        return 1
    rec = {"frozen_utc": utc(), "registration": {os.path.relpath(reg, STUDY): sha256(reg)},
           "src": tree_hashes(os.path.join(STUDY, "src")),
           "extra": {os.path.relpath(os.path.abspath(p), STUDY): sha256(p) for p in argv[1:]}}
    with open(out, "w") as f:
        json.dump(rec, f, indent=1, sort_keys=True)
    for p in (reg, out):
        os.chmod(p, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    print(json.dumps({"frozen_utc": rec["frozen_utc"], "registration_sha256": sha256(reg)}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
