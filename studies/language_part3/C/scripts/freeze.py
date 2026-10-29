"""Freeze the C-core registration: write R1_C-core.frozen.json with the UTC time, the SHA-256 of the
registration text and of every tracked file (this arm's code, the imported tally code, since the
step-9 amendment dl tm.py and cpu_tm_fast.py, the PTB train and validation text, the C4 word table,
T0's 30 frozen output folders), plus the recorded, not enforced, hashes of the plan, proposal C, the
CPU gate script and dl tm_init.py. Real stages refuse to start without this record, when a tracked
file changed or is not in it, or when LOG.md lacks its SHA-256 (ccore/hashing.py check_frozen).

    nice -n 19 .venv-tm/bin/python -I -B scripts/freeze.py                      # the first freeze (done)
    nice -n 19 .venv-tm/bin/python -I -B scripts/freeze.py --amend "<reason>"   # a dated amendment

--amend (code review of 2026-10-21, issue 7, as A-Struct's freeze.py): the current record is kept
byte for byte as history/R1_C-core.frozen.v<N>.json (never overwritten), then a new record is written
with the same fields plus the amendment's reason and number, the previous record's SHA-256, UTC time
and copy, and the finished real runs at the freeze. Its SHA-256 (printed) must go into a dated
amendment line in LOG.md before any real stage runs. Without --amend an existing record is never
overwritten."""
import argparse
import datetime
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ccore import hashing, paths, runlog  # noqa: E402


def real_runs():
    """Names of the finished real (not dry) run folders."""
    out = []
    for d in sorted(glob.glob(str(paths.RUNS / "*"))):
        if not os.path.isdir(d) or d.endswith("_dry") or not os.path.exists(os.path.join(d, "run.json")):
            continue
        try:
            if runlog.read_json(os.path.join(d, "run.json")).get("state") == "finished":
                out.append(os.path.basename(d))
        except (OSError, ValueError):
            pass
    return out


def keep_previous():
    """Copy the current record byte for byte into history/ and describe it (for --amend)."""
    old = paths.FROZEN.read_bytes()
    old_rec = json.loads(old)
    hist = paths.FROZEN.parent / "history"
    os.makedirs(hist, exist_ok=True)
    n = 1 + len(glob.glob(str(hist / (paths.FROZEN.stem + ".v*.json"))))
    copy = hist / f"{paths.FROZEN.stem}.v{n}.json"
    with open(copy, "xb") as f:                 # never overwrites
        f.write(old)
    if hashing.sha256(copy) != hashing.sha256_bytes(old):
        raise SystemExit(f"the copy {copy} differs from the record")
    return n, {"sha256": hashing.sha256_bytes(old), "frozen_utc": old_rec.get("frozen_utc"),
               "amendment": old_rec.get("amendment"), "copy": hashing.rel(copy)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--amend", default=None,
                    help="reason of a dated amendment (the existing record is kept in history/)")
    a = ap.parse_args(argv)
    if not paths.REG.exists():
        raise SystemExit(f"no registration text: {paths.REG}")
    if a.amend is not None and not a.amend.strip():
        raise SystemExit("--amend needs a reason")
    if paths.FROZEN.exists() and a.amend is None:
        raise SystemExit(f"already frozen: {paths.FROZEN}; a later change needs --amend \"<reason>\" and a dated "
                         "LOG.md line")
    if not paths.FROZEN.exists() and a.amend is not None:
        raise SystemExit(f"nothing to amend: no frozen record {paths.FROZEN}")
    files = hashing.snapshot()
    number, prev = keep_previous() if a.amend is not None else (None, None)
    rec = {"registration": hashing.rel(paths.REG), "registration_sha256": hashing.sha256(paths.REG),
           "frozen_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "python": sys.version.split()[0], "files": files,
           "reference_files_not_enforced": hashing.reference_snapshot()}
    if a.amend is not None:
        rec.update({"amendment": a.amend, "amendment_number": number, "previous_record": prev,
                    "real_runs_at_freeze": real_runs()})
    tmp = str(paths.FROZEN) + ".part"
    with open(tmp, "w") as f:
        json.dump(rec, f, indent=1, sort_keys=True)
    os.replace(tmp, paths.FROZEN)
    code = {k: v for k, v in files.items() if "/C/" in k}
    print(rec["frozen_utc"], "registration", rec["registration_sha256"], "files", len(files), "of them C code", len(code))
    print(f"{hashing.sha256(paths.FROZEN)}  {hashing.rel(paths.FROZEN)}")
    if prev is not None:
        print(f"amendment {number}: previous record {prev['sha256']} kept as {prev['copy']}")


if __name__ == "__main__":
    main()
