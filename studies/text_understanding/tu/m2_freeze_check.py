#!/usr/bin/env python
"""M2 freeze check (tu/PLAN_M2.md "Freeze procedure"): the literal comparison between the registration's pins block
and the pins file the chain checks before every step. Added after the review of draft 1 (design MAJOR 2: a folder
checked only against its own manifest proves nothing).

    python -I -B m2_freeze_check.py --plan tu/PLAN_M2.md --pins tu/m2/pins.sh [--dry tu/m2_dry/<UTC>/dry_summary.json]

Refuses (exit 1, every reason printed) unless all of these hold:
1. the plan's "## Pins" section (from that heading line to the next "## " heading line) holds a sha256sum block (lines
   "<64 hex>  <path>"; a relative path in the block is read from glassbox-lm/studies/text_understanding); every one of those lines equals
   its file now, and no path appears there with two different sha256 (a stale line is refused wherever it stands);
2. every "check <file> <sha256>" line of the pins file names an absolute path (the chain reads the pins from its own
   working directory) and equals its file now, and every one of them except the plan itself appears in the block with
   the same full sha256;
3. the pins file pins the plan itself, with the plan's current sha256 (the chain then stops if the registration
   changes);
4. with --dry: the dry run's summary says "pass": true, and its "code_sha256" (the sha256 of every file of
   m2_write_pins.sh --list, read when the dry run started) names every pinned file except the plan and the summary
   itself, with the same sha256 as now (the plan may change between the dry run and the freeze, so it is left out).
The paths given on the command line are read from the current working directory. Prints "freeze check ok: ..." on
success. Reads only the files it is given and the files they name.
Changes after the re-check of draft 2 (2026-10-25, MINOR C and D): every block line is kept (before, only the last line
per path counted, so a stale line placed before the right one passed); command-line paths are read from the working
directory (before, a relative --plan was read from glassbox-lm/studies/text_understanding); relative pins lines are refused.
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

EMB = Path(f"{_REPO_ROOT}/studies/text_understanding")
LINE = re.compile(r"^([0-9a-f]{64})  (\S.*)$")


def sha256(p: str) -> str | None:
    try:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        return h.hexdigest()
    except OSError:
        return None


def norm(p: str) -> str:
    """A block path: absolute as given, or relative to glassbox-lm/studies/text_understanding."""
    q = Path(p)
    return os.path.normpath(str(q if q.is_absolute() else EMB / q))


def plan_block(plan: Path) -> list[tuple[str, str]]:
    """Every (path, sha256) line of the plan's "## Pins" section, in order, duplicates kept."""
    lines, inside = [], False
    for ln in plan.read_text().splitlines():
        if ln.startswith("## "):
            inside = ln.startswith("## Pins")
            continue
        m = LINE.match(ln) if inside else None
        if m:
            lines.append((norm(m.group(2)), m.group(1)))
    return lines


def pins_lines(pins: Path) -> tuple[dict[str, str], list[str]]:
    out, rel = {}, []
    for ln in pins.read_text().splitlines():
        parts = ln.split()
        if len(parts) == 3 and parts[0] == "check":
            if not Path(parts[1]).is_absolute():
                rel.append(parts[1])
            out[os.path.normpath(os.path.abspath(parts[1]))] = parts[2]
    return out, rel


def check(plan: Path, pins: Path, dry: Path | None) -> tuple[list[str], str]:
    errs = []
    lines = plan_block(plan)
    if not lines:
        return ["the plan has no sha256sum block under '## Pins'"], ""
    block: dict[str, set[str]] = {}
    for f, h in lines:
        block.setdefault(f, set()).add(h)
    for f, hs in block.items():
        if len(hs) > 1:
            errs.append(f"the plan's block names this file with {len(hs)} different sha256: {f}")
        now = sha256(f)
        for h in sorted(hs):
            if now != h:
                errs.append(f"block line differs from the file now: {f} (block {h[:16]}, now {(now or 'missing')[:16]})")
    pl, rel = pins_lines(pins)
    if not pl:
        errs.append(f"{pins} holds no check lines")
    for r in rel:
        errs.append(f"a pins line with a relative path (the chain would read it from its working directory): {r}")
    me = os.path.normpath(os.path.abspath(plan))
    for f, h in pl.items():
        now = sha256(f)
        if len(h) != 64 or now != h:
            errs.append(f"pins line differs from the file now (or is not a full sha256): {f}")
        if f != me and block.get(f) != {h}:
            errs.append(f"pinned file not in the plan's block with the same sha256: {f}")
    if pl.get(me) != sha256(me):
        errs.append(f"the pins file does not pin the plan itself with its current sha256: {me}")
    if dry is not None:
        try:
            s = json.loads(dry.read_text())
        except (OSError, ValueError) as e:
            s = {}
            errs.append(f"dry summary unreadable: {e}")
        if s.get("pass") is not True:
            errs.append("the dry summary does not say pass: true")
        cs = {norm(k): v for k, v in (s.get("code_sha256") or {}).items()}
        dme = os.path.normpath(os.path.abspath(dry))
        for f in pl:
            if f in (me, dme):
                continue
            if f not in cs:
                errs.append(f"the dry run did not record this pinned file: {f}")
            elif cs[f] != sha256(f):
                errs.append(f"the file changed after the dry run: {f}")
    msg = f"freeze check ok: {len(pl)} pins, {len(lines)} block lines" + (", dry summary ok" if dry is not None else "")
    return errs, msg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--pins", type=Path, required=True)
    ap.add_argument("--dry", type=Path, default=None)
    a = ap.parse_args()
    errs, msg = check(a.plan, a.pins, a.dry)
    if errs:
        print("freeze check REFUSED:\n" + "\n".join(errs))
        return 1
    print(msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
