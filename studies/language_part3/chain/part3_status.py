#!/usr/bin/env python
"""Helpers of chain/part3_cpu_job.sh (no writes; run with python -I -B).

    part3_status.py diff-class <diff file>             prints none | parent-only | changed
    part3_status.py c-decision <decision json> <epoch>  prints C's decision; exit 0 = PASS, 3 = STOP, 1 = missing,
                                                        older than <epoch> (not written by this job) or unknown

diff-class reads the output of `diff before after` of C's runner listings (`ls -la --time-style=full-iso`, one block
per imported folder): every changed line ("< " or "> ") whose last field is ".." describes a parent folder, which
other sessions change by writing next to it; "parent-only" means no other line changed.
"""

import json
import os
import sys


def diff_class(text: str) -> str:
    changed = [ln[2:] for ln in text.splitlines() if ln.startswith(("< ", "> "))]
    if not changed:
        return "none"
    return "parent-only" if all(ln.split()[-1:] == [".."] for ln in changed) else "changed"


def c_decision(path: str, not_before: float) -> tuple:
    if not os.path.exists(path):
        return 1, f"missing: {path}"
    if os.path.getmtime(path) < not_before:
        return 1, f"older than this job: {path}"
    with open(path) as f:
        d = json.load(f).get("decision", "")
    if d.startswith("STOP"):
        return 3, d
    if d.startswith("PASS"):
        return 0, d
    return 1, f"unknown decision: {d!r}"


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "diff-class":
        p = sys.argv[2]
        print(diff_class(open(p).read()) if os.path.exists(p) else "missing")
        return 0
    if len(sys.argv) == 4 and sys.argv[1] == "c-decision":
        rc, msg = c_decision(sys.argv[2], float(sys.argv[3]))
        print(msg)
        return rc
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
