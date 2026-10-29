#!/usr/bin/env python
"""Before/after listings of the imported folders (PLAN_part3.md section 6): ls -la --time-style=full-iso of each
folder in astruct.paths.LISTED, never recursive.

    listing.py snapshot <file>        # write the listing
    listing.py compare <before> <after>   # "no change" (exit 0) or the changed lines (exit 1)

Every line of a folder's own listing is compared, "." included; a ".." line describes the parent folder, which other
sessions may change, so a ".." difference is printed as a note and does not fail the check.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from astruct.paths import LISTED


def snapshot(folders=None) -> str:
    out = []
    for d in folders or LISTED:
        out.append(f"### {d}")
        if Path(d).is_dir():
            r = subprocess.run(["ls", "-la", "--time-style=full-iso", str(d)], capture_output=True, text=True, check=True)
            out.append(r.stdout.rstrip("\n"))
        else:
            out.append("(absent)")
    return "\n".join(out) + "\n"


def _parse(text: str) -> dict:
    sec, cur = {}, None
    for line in text.splitlines():
        if line.startswith("### "):
            cur = line[4:]
            sec[cur] = []
        elif cur is not None:
            sec[cur].append(line)
    return sec


def _is_parent(line: str) -> bool:
    return line.split()[-1:] == [".."]


def compare(before: str, after: str) -> tuple[bool, list, list]:
    a, b = _parse(before), _parse(after)
    changes, notes = [], []
    for d in sorted(set(a) | set(b)):
        la, lb = a.get(d, ["(missing section)"]), b.get(d, ["(missing section)"])
        pa = [x for x in la if _is_parent(x)]
        pb = [x for x in lb if _is_parent(x)]
        ra = [x for x in la if not _is_parent(x)]
        rb = [x for x in lb if not _is_parent(x)]
        if ra != rb:
            for x in sorted(set(ra) - set(rb)):
                changes.append(f"{d}: - {x}")
            for x in sorted(set(rb) - set(ra)):
                changes.append(f"{d}: + {x}")
        if pa != pb:
            notes.append(f"{d}: parent entry changed (not counted): {pa} -> {pb}")
    return not changes, changes, notes


def main() -> int:
    if len(sys.argv) >= 3 and sys.argv[1] == "snapshot":
        Path(sys.argv[2]).write_text(snapshot())
        return 0
    if len(sys.argv) >= 4 and sys.argv[1] == "compare":
        ok, ch, notes = compare(Path(sys.argv[2]).read_text(), Path(sys.argv[3]).read_text())
        for n in notes:
            print("note:", n)
        if ok:
            print("listing: no change in", len(_parse(Path(sys.argv[2]).read_text())), "folders")
            return 0
        print("listing: CHANGE")
        for c in ch:
            print(" ", c)
        return 1
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
