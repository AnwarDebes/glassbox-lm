"""Freeze a plan of this research line: write <plan>.frozen.json with the UTC time read from the clock, the
SHA-256 of the plan text, of every file under src/ and tests/, and of any extra file named on the command line
(recorded with its path relative to the study folder).
Refuses to overwrite an existing record.

Usage (from the study folder):  .venv/bin/python -I src/freeze.py registrations/<name>.md [extra files ...]
Check later:                    .venv/bin/python -I src/freeze.py --check registrations/<name>.frozen.json
The check exits with status 1 when the plan, a source file or a pinned file differs from the record, or when a
source file was added or removed.
"""
import datetime
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tree(*folders):
    out = {}
    for folder in folders:
        base = os.path.join(ROOT, folder)
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
            for name in sorted(filenames):
                if name.endswith((".py", ".sh", ".md", ".txt", ".json")):
                    p = os.path.join(dirpath, name)
                    out[os.path.relpath(p, ROOT)] = sha(p)
    return out


def freeze(plan, extras):
    plan = os.path.abspath(plan)
    record = os.path.splitext(plan)[0] + ".frozen.json"
    if os.path.exists(record):
        raise SystemExit(f"refusing to overwrite {record}")
    rec = {
        "plan": os.path.relpath(plan, ROOT),
        "plan_sha256": sha(plan),
        "frozen_utc": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sources_sha256": tree("src", "tests"),
        "extras_sha256": {os.path.relpath(os.path.abspath(e), ROOT): sha(e) for e in extras},
    }
    with open(record, "x") as f:
        json.dump(rec, f, indent=1)
    os.chmod(record, 0o444)
    os.chmod(plan, 0o444)
    print("frozen", rec["frozen_utc"], rec["plan_sha256"], "->", os.path.relpath(record, ROOT))
    return rec


def check(record):
    with open(record) as f:
        rec = json.load(f)
    now_plan = sha(os.path.join(ROOT, rec["plan"]))
    cur = tree("src", "tests")
    changed = sorted(k for k in rec["sources_sha256"] if cur.get(k) not in (None, rec["sources_sha256"][k]))
    gone = sorted(k for k in rec["sources_sha256"] if k not in cur)
    new = sorted(k for k in cur if k not in rec["sources_sha256"])
    extras_bad = sorted(k for k, v in rec["extras_sha256"].items()
                        if not os.path.exists(os.path.join(ROOT, k)) or sha(os.path.join(ROOT, k)) != v)
    out = {"plan_unchanged": now_plan == rec["plan_sha256"], "changed": changed, "gone": gone, "new": new,
           "extras_changed": extras_bad, "frozen_utc": rec["frozen_utc"]}
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    if args[0] == "--check":
        res = check(args[1])
        sys.exit(0 if res["plan_unchanged"] and not (res["extras_changed"] or res["changed"] or res["gone"] or res["new"]) else 1)
    freeze(args[0], args[1:])
