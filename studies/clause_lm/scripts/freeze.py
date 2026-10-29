"""Freeze a plan or registration of clause_lm: write <name>.frozen.json beside it.

    .venv-cdllm/bin/python scripts/freeze.py [--check] docs/<plan>.md

Run from studies/clause_lm (added 2026-10-25; the study's earlier registration, docs/prereg_M5_step1.md, had no freeze
record). The record holds the UTC time read from the clock (frozen_utc), the SHA-256 of the plan text, of every file
under src/, scripts/ and tests/ of the study (.py, .sh, .patch), and of every file that the plan's hash blocks pin. A
hash block is a fenced block of sha256sum lines ("<64 hex>  <path>") under a heading that starts with "## Hashes"; a
relative path is read from the study folder. Every pinned file is hashed again: when one is missing or differs from
the plan's value, nothing is written and the exit code is 1, so the pins of a frozen plan held at its freeze. A plan
without a hash block is refused too. The tool never overwrites a record (exit 1). It makes the plan and the record
read-only; the mode is not part of the record (a copy or a checkout resets it), the hashes are. A frozen plan is never
edited; a change is a dated note in a new file. With --check the tool hashes the pins, prints the result and writes
nothing (exit 0 when every pin holds, 1 otherwise), for a check before the freeze. Exit 2: usage error.
"""
import datetime
import hashlib
import json
import os
import re
import stat
import sys

STUDY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USAGE = "usage: .venv-cdllm/bin/python scripts/freeze.py [--check] docs/<plan>.md"
HASH_LINE = re.compile(r"^([0-9a-f]{64})  (\S.*)$")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def plan_pins(text):
    """[(sha256, path)] of the sha256sum lines in fenced blocks under the '## Hashes' headings of a plan."""
    pins, heading, fenced = [], "", False
    for line in text.splitlines():
        if line.startswith("```"):
            fenced = not fenced
            continue
        if not fenced and line.startswith("## "):
            heading = line
            continue
        if fenced and heading.startswith("## Hashes"):
            m = HASH_LINE.match(line)
            if m:
                pins.append((m.group(1), m.group(2)))
    return pins


def study_files():
    out = {}
    for d in ("src", "scripts", "tests"):
        for root, dirs, files in os.walk(os.path.join(STUDY, d)):
            dirs[:] = sorted(x for x in dirs if x != "__pycache__")
            for f in sorted(files):
                if f.endswith((".py", ".sh", ".patch")):
                    p = os.path.join(root, f)
                    out[os.path.relpath(p, STUDY)] = sha256(p)
    return out


def main(argv):
    check_only = bool(argv) and argv[0] == "--check"
    if check_only:
        argv = argv[1:]
    if len(argv) != 1 or argv[0] in ("-h", "--help") or not argv[0].endswith(".md"):
        print(USAGE, file=sys.stderr)
        return 2
    plan = os.path.abspath(argv[0])
    if not os.path.isfile(plan):
        print(f"not a file: {argv[0]}\n{USAGE}", file=sys.stderr)
        return 2
    out = plan[:-3] + ".frozen.json"
    if os.path.exists(out) and not check_only:
        print(f"already frozen: {out}", file=sys.stderr)
        return 1
    pins = plan_pins(open(plan, encoding="utf8").read())
    if not pins:
        print("refused: the plan has no hash block (a fenced block of sha256sum lines under '## Hashes')", file=sys.stderr)
        return 1
    pinned, bad = {}, []
    for want, p in pins:
        full = p if os.path.isabs(p) else os.path.join(STUDY, p)
        got = sha256(full) if os.path.isfile(full) else None
        if got != want:
            bad.append(f"{p}: plan {want}, file {got or 'missing'}")
        pinned[p] = want
    if bad:
        print("refused: pinned files differ from the plan:\n  " + "\n  ".join(bad), file=sys.stderr)
        return 1
    if check_only:
        print(json.dumps({"check": "every pin holds", "pinned": len(pinned), "plan_sha256": sha256(plan),
                          "already_frozen": os.path.exists(out)}))
        return 0
    rec = {"frozen_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "plan": {os.path.relpath(plan, STUDY): sha256(plan)}, "pinned": pinned, "study_files": study_files(),
           "tool": {"scripts/freeze.py": sha256(os.path.abspath(__file__))}}
    with open(out, "x") as f:
        json.dump(rec, f, indent=1, sort_keys=True)
    for p in (plan, out):
        os.chmod(p, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    print(json.dumps({"frozen_utc": rec["frozen_utc"], "plan_sha256": sha256(plan), "pinned": len(pinned),
                      "record": os.path.relpath(out, STUDY)}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
