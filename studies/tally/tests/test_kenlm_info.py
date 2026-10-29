"""floor/kenlm.py info(): what is recorded about the KenLM binaries.

Run: PYTHONPATH=src python tests/test_kenlm_info.py

Needs the KenLM binaries (tools/kenlm/bin of the workspace or KENLM_BIN). Checks that the
sha256 values equal a fresh hash of each binary, that the maximum order printed by query is
read (at least 5, the highest order A1 builds), and that a stated source commit, when one is
found, is a full 40-digit hash.
"""
import hashlib
import os
import re
import sys

from tally.floor import kenlm

FAILS = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f" ({detail})" if detail and not cond else ""), flush=True)
    if not cond:
        FAILS.append(name)


def main():
    i = kenlm.info()
    print("info:", i)
    for b in ("lmplz", "build_binary", "query"):
        p = os.path.join(kenlm.BIN, b)
        want = hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None
        check(f"sha256 of {b} equals a fresh hash", want is not None and i["sha256"].get(b) == want)
    check("maximum order read from query (>= 5)", isinstance(i["max_order"], int) and i["max_order"] >= 5,
          f"{i['max_order']}")
    c = i["source_commit_stated"]
    check("stated source commit is a 40-digit hash or absent with no path",
          (c is None and i["stated_in"] is None) or (re.fullmatch(r"[0-9a-f]{40}", c or "") and os.path.exists(i["stated_in"])))
    print("PASS" if not FAILS else f"FAIL: {len(FAILS)} checks")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
